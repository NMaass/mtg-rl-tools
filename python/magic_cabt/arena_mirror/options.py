"""Arena decision prompts as CABT-style indexed option lists.

Every GRE decision prompt becomes ``{"select": {"type", "minCount",
"maxCount", "option": [{"index", "type", "label", "payload"}]}}`` — the same
shape the Java bridge serializes for engine decisions — and the client's
response message is matched back to option indices. Responses carry
``respId`` equal to the prompt's ``msgId``, so pairing is exact.

Matching is best-effort by design: a record whose response cannot be mapped
to indices keeps the raw payload and is flagged ``selectionMatched: false``
rather than dropped.
"""

__all__ = ["build_prompt", "match_response"]


def build_prompt(event):
    """Normalized ARENA_DECISION_PROMPT event -> prompt dict, or None."""
    payload = event.get("payload") or {}
    message_type = event.get("messageType") or payload.get("type") or ""
    kind = message_type.replace("GREMessageType_", "")
    builder = _PROMPT_BUILDERS.get(kind, _build_generic)
    min_count, max_count, options = builder(payload)
    return {
        "messageType": message_type,
        "kind": kind,
        "msgId": payload.get("msgId"),
        "gameStateId": payload.get("gameStateId"),
        "timestamp": event.get("timestamp"),
        "select": {
            "type": kind.upper(),
            "minCount": min_count,
            "maxCount": max_count,
            "option": [
                {"index": i, "type": option[0], "label": option[1],
                 "payload": option[2]}
                for i, option in enumerate(options)
            ],
        },
    }


def match_response(pending_prompts, event):
    """Pair an ARENA_CLIENT_DECISION event with its prompt.

    Returns ``(prompt, selected_indices_or_None, matched_bool)`` or None when
    the message is not a decision to record (e.g. a bare Submit confirmation
    with no matching prompt).
    """
    payload = event.get("payload") or {}
    message_type = event.get("messageType") or payload.get("type") or ""
    kind = message_type.replace("ClientMessageType_", "")

    if kind == "ConcedeReq":
        prompt = _concede_prompt(event)
        return prompt, [0], True

    resp_id = payload.get("respId")
    prompt = None
    if resp_id is not None:
        for candidate in pending_prompts:
            if candidate.get("msgId") == resp_id:
                prompt = candidate
                break
    if prompt is None:
        # Submit* confirmations ride behind their Resp; nothing to record.
        return None

    matcher = _RESPONSE_MATCHERS.get(kind)
    if matcher is None:
        return prompt, None, False
    selected = matcher(prompt, payload)
    if selected is None:
        return prompt, None, False
    # dedupe but preserve order: for ORDER-type decisions the sequence of
    # indices IS the chosen ordering
    seen = set()
    ordered = []
    for index in selected:
        if index not in seen:
            seen.add(index)
            ordered.append(index)
    return prompt, ordered, True


# --- prompt builders: payload -> (min_count, max_count, [(type, label, payload)]) ---

def _build_actions_available(payload):
    request = payload.get("actionsAvailableReq") or {}
    options = []
    for action in request.get("actions") or []:
        action_type = (action.get("actionType") or "").replace("ActionType_", "")
        label = action_type
        if action.get("grpId"):
            label = "%s grpId=%s instance=%s" % (
                action_type, action.get("grpId"), action.get("instanceId"))
        options.append((action_type.upper() or "ACTION", label, action))
    return 1, 1, options


def _build_choose_starting_player(payload):
    request = payload.get("chooseStartingPlayerReq") or {}
    options = []
    for seat_id in request.get("systemSeatIds") or []:
        options.append(("SEAT", "seat %s plays first" % seat_id,
                        {"systemSeatId": seat_id}))
    return 1, 1, options


def _build_mulligan(payload):
    request = payload.get("mulliganReq") or {}
    options = [
        ("KEEP", "keep hand", {"decision": "MulliganOption_AcceptHand"}),
        ("MULLIGAN", "mulligan", {"decision": "MulliganOption_Mulligan"}),
    ]
    return 1, 1, options


def _build_select_targets(payload):
    request = payload.get("selectTargetsReq") or {}
    options = []
    min_count = 0
    max_count = 0
    for group in request.get("targets") or []:
        target_idx = group.get("targetIdx")
        min_count += group.get("minTargets") or 0
        max_count += group.get("maxTargets") or 0
        for target in group.get("targets") or []:
            instance_id = target.get("targetInstanceId")
            options.append(("TARGET",
                            "target instance=%s (group %s)" % (instance_id, target_idx),
                            {"targetIdx": target_idx,
                             "targetInstanceId": instance_id,
                             "legalAction": target.get("legalAction")}))
    return min_count, max_count or len(options), options


def _build_declare_attackers(payload):
    request = payload.get("declareAttackersReq") or {}
    attackers = request.get("qualifiedAttackers") or request.get("attackers") or []
    options = []
    for attacker in attackers:
        attacker_id = attacker.get("attackerInstanceId")
        for recipient in attacker.get("legalDamageRecipients") or [{}]:
            label = "attack with instance=%s -> %s" % (
                attacker_id, _recipient_label(recipient))
            options.append(("ATTACK", label,
                            {"attackerInstanceId": attacker_id,
                             "recipient": recipient}))
    return 0, len(options), options


def _build_declare_blockers(payload):
    request = payload.get("declareBlockersReq") or {}
    options = []
    for blocker in request.get("blockers") or []:
        blocker_id = blocker.get("blockerInstanceId")
        for attacker_id in blocker.get("attackerInstanceIds") or []:
            options.append(("BLOCK",
                            "block attacker instance=%s with instance=%s"
                            % (attacker_id, blocker_id),
                            {"blockerInstanceId": blocker_id,
                             "attackerInstanceId": attacker_id}))
    return 0, len(options), options


def _build_casting_time_options(payload):
    request = payload.get("castingTimeOptionsReq") or {}
    options = []
    min_count = 1
    max_count = 1
    for cto in request.get("castingTimeOptionReq") or []:
        modal = cto.get("modalReq")
        if isinstance(modal, dict):
            min_count = modal.get("minSel", min_count)
            max_count = modal.get("maxSel", max_count)
            for modal_option in modal.get("modalOptions") or []:
                options.append(("MODE",
                                "mode grpId=%s" % modal_option.get("grpId"),
                                {"ctoId": cto.get("ctoId"),
                                 "grpId": modal_option.get("grpId")}))
        else:
            options.append(("CASTING_OPTION",
                            (cto.get("castingTimeOptionType") or "option"),
                            cto))
    return min_count, max_count, options


def _build_pay_costs(payload):
    request = payload.get("payCostsReq") or {}
    payment = (request.get("paymentActions") or {}).get("actions") or []
    options = []
    for action in payment:
        options.append(("PAY",
                        "%s instance=%s" % (
                            (action.get("actionType") or "").replace("ActionType_", ""),
                            action.get("instanceId")),
                        action))
    auto_tap = request.get("autoTapActionsReq")
    if auto_tap is not None:
        options.append(("AUTO_TAP", "auto-tap suggested payment", auto_tap))
    # Non-mana effect costs ("sacrifice N creatures", ...) come as a
    # selection list instead of payment actions.
    effect_cost = request.get("effectCostReq") or {}
    selection = effect_cost.get("costSelection") or {}
    min_count = selection.get("minSel") or 0
    for candidate_id in selection.get("ids") or []:
        options.append(("COST_SELECT", "pay with id=%s" % candidate_id,
                        {"id": candidate_id}))
    return min_count, max(len(options), 1), options


def _build_search(payload):
    request = payload.get("searchReq") or {}
    options = []
    for instance_id in request.get("itemsSought") or request.get("itemsToSearch") or []:
        options.append(("FIND", "find instance=%s" % instance_id,
                        {"instanceId": instance_id}))
    if (request.get("allowFailToFind") or "").endswith(("_Any", "_Yes")):
        options.append(("FAIL_TO_FIND", "fail to find", {}))
    return 0, request.get("maxFind") or 1, options


def _build_select_n(payload):
    request = payload.get("selectNReq") or {}
    options = []
    for instance_id in request.get("ids") or []:
        options.append(("SELECT", "select id=%s" % instance_id,
                        {"id": instance_id}))
    return request.get("minSel") or 0, request.get("maxSel") or len(options), options


def _build_group(payload):
    request = payload.get("groupReq") or {}
    specs = request.get("groupSpecs") or []
    spec_note = " / ".join(
        "%s->%s" % (spec.get("zoneType"), spec.get("subZoneType"))
        for spec in specs if isinstance(spec, dict))
    options = []
    for instance_id in request.get("instanceIds") or []:
        options.append(("GROUP_ITEM",
                        "id=%s (%s)" % (instance_id, spec_note),
                        {"id": instance_id}))
    return 0, len(options), options


def _build_numeric(payload):
    request = payload.get("numericInputReq") or {}
    return 1, 1, [("NUMBER", "numeric input", request)]


def _build_optional_action(payload):
    return 1, 1, [
        ("YES", "accept optional action", {"answer": "yes"}),
        ("NO", "decline optional action", {"answer": "no"}),
    ]


def _build_order(payload):
    request = payload.get("orderReq") or {}
    options = []
    for instance_id in request.get("ids") or request.get("instanceIds") or []:
        options.append(("ORDER_ITEM", "id=%s" % instance_id, {"id": instance_id}))
    return len(options), len(options), options


def _build_assign_damage(payload):
    request = payload.get("assignDamageReq") or {}
    options = []
    for assigner in request.get("damageAssigners") or []:
        assigner_id = assigner.get("instanceId")
        for assignment in assigner.get("assignments") or []:
            options.append(("ASSIGN_DAMAGE",
                            "instance=%s damages instance=%s (min %s)"
                            % (assigner_id, assignment.get("instanceId"),
                               assignment.get("minDamage")),
                            {"assignerInstanceId": assigner_id,
                             "instanceId": assignment.get("instanceId"),
                             "minDamage": assignment.get("minDamage")}))
    return 0, max(len(options), 1), options


def _build_generic(payload):
    return 0, 0, []


_PROMPT_BUILDERS = {
    "ActionsAvailableReq": _build_actions_available,
    "ChooseStartingPlayerReq": _build_choose_starting_player,
    "MulliganReq": _build_mulligan,
    "SelectTargetsReq": _build_select_targets,
    "DeclareAttackersReq": _build_declare_attackers,
    "DeclareBlockersReq": _build_declare_blockers,
    "CastingTimeOptionsReq": _build_casting_time_options,
    "PayCostsReq": _build_pay_costs,
    "SearchReq": _build_search,
    "SelectNReq": _build_select_n,
    "GroupReq": _build_group,
    "NumericInputReq": _build_numeric,
    "OptionalActionMessage": _build_optional_action,
    "OrderReq": _build_order,
    "AssignDamageReq": _build_assign_damage,
}


# --- response matchers: (prompt, response_payload) -> [indices] or None ---

def _options(prompt):
    return prompt["select"]["option"]


def _match_perform_action(prompt, payload):
    response_actions = (payload.get("performActionResp") or {}).get("actions") or []
    indices = []
    for action in response_actions:
        found = _find_option(prompt, lambda option: (
            option["payload"].get("actionType") == action.get("actionType")
            and option["payload"].get("instanceId") == action.get("instanceId")
            and option["payload"].get("grpId") == action.get("grpId")))
        if found is None:
            # Arena sometimes answers with an inactiveAction (alternative
            # cost) or an action the prompt never listed; fall back to
            # actionType+instanceId, then give up.
            found = _find_option(prompt, lambda option: (
                option["payload"].get("actionType") == action.get("actionType")
                and option["payload"].get("instanceId") == action.get("instanceId")))
        if found is None:
            return None
        indices.append(found)
    return indices


def _match_choose_starting_player(prompt, payload):
    response = payload.get("chooseStartingPlayerResp") or {}
    seat_id = response.get("systemSeatId")
    found = _find_option(prompt, lambda option:
                         option["payload"].get("systemSeatId") == seat_id)
    return None if found is None else [found]


def _match_mulligan(prompt, payload):
    decision = (payload.get("mulliganResp") or {}).get("decision")
    found = _find_option(prompt, lambda option:
                         option["payload"].get("decision") == decision)
    return None if found is None else [found]


def _match_select_targets(prompt, payload):
    target_group = (payload.get("selectTargetsResp") or {}).get("target") or {}
    target_idx = target_group.get("targetIdx")
    indices = []
    for target in target_group.get("targets") or []:
        found = _find_option(prompt, lambda option: (
            option["payload"].get("targetIdx") == target_idx
            and option["payload"].get("targetInstanceId")
            == target.get("targetInstanceId")))
        if found is None:
            return None
        indices.append(found)
    return indices


def _match_declare_attackers(prompt, payload):
    response = payload.get("declareAttackersResp") or {}
    if response.get("autoDeclare"):
        recipient = response.get("autoDeclareDamageRecipient")
        indices = []
        seen_attackers = set()
        for i, option in enumerate(_options(prompt)):
            attacker = option["payload"].get("attackerInstanceId")
            if attacker in seen_attackers:
                continue
            if recipient is None or option["payload"].get("recipient") == recipient:
                indices.append(i)
                seen_attackers.add(attacker)
        return indices
    indices = []
    for attacker in response.get("selectedAttackers") or []:
        attacker_id = attacker.get("attackerInstanceId")
        recipient = attacker.get("selectedDamageRecipient") \
            or attacker.get("damageRecipient")
        found = _find_option(prompt, lambda option: (
            option["payload"].get("attackerInstanceId") == attacker_id
            and (recipient is None
                 or option["payload"].get("recipient") == recipient)))
        if found is None:
            return None
        indices.append(found)
    return indices


def _match_declare_blockers(prompt, payload):
    response = payload.get("declareBlockersResp") or {}
    indices = []
    for blocker in response.get("selectedBlockers") or []:
        blocker_id = blocker.get("blockerInstanceId")
        for attacker_id in blocker.get("selectedAttackerInstanceIds") or []:
            found = _find_option(prompt, lambda option: (
                option["payload"].get("blockerInstanceId") == blocker_id
                and option["payload"].get("attackerInstanceId") == attacker_id))
            if found is None:
                return None
            indices.append(found)
    return indices


def _match_casting_time_options(prompt, payload):
    response = (payload.get("castingTimeOptionsResp") or {}) \
        .get("castingTimeOptionResp") or {}
    modal = response.get("chooseModalResp")
    if isinstance(modal, dict):
        indices = []
        for grp_id in modal.get("grpIds") or []:
            found = _find_option(prompt, lambda option:
                                 option["payload"].get("grpId") == grp_id)
            if found is None:
                return None
            indices.append(found)
        return indices
    # Non-modal casting-time choice (kicker, optional cost, done): the
    # response names the chosen ctoId + option type.
    cto_id = response.get("ctoId")
    cto_type = response.get("castingTimeOptionType")
    if cto_type is not None:
        found = _find_option(prompt, lambda option: (
            option["payload"].get("castingTimeOptionType") == cto_type
            and (cto_id is None
                 or option["payload"].get("ctoId") in (cto_id, None))))
        if found is not None:
            return [found]
    return None


def _match_search(prompt, payload):
    response = payload.get("searchResp") or {}
    items = response.get("itemsFound")
    if not items:
        found = _find_option(prompt, lambda option: option["type"] == "FAIL_TO_FIND")
        return None if found is None else [found]
    indices = []
    for instance_id in items:
        found = _find_option(prompt, lambda option:
                             option["payload"].get("instanceId") == instance_id)
        if found is None:
            return None
        indices.append(found)
    return indices


def _match_select_n(prompt, payload):
    response = payload.get("selectNResp") or {}
    option_count = len(_options(prompt))
    selected_ids = response.get("ids") or []
    indices = []
    for selected_id in selected_ids:
        found = _find_option(prompt, lambda option:
                             option["payload"].get("id") == selected_id)
        if found is None:
            indices = None
            break
        indices.append(found)
    if indices is not None:
        return indices
    if response.get("useArbitrary"):
        # "Any order is fine": the client answers with positional ids and an
        # arbitrary-ordering marker; record the options in given order.
        if all(isinstance(i, int) and 0 <= i < option_count
               for i in selected_ids):
            return selected_ids if selected_ids != [0] or option_count == 1 \
                else list(range(option_count))
        return list(range(option_count))
    return None


def _match_group(prompt, payload):
    # Ordered grouping (e.g. London mulligan bottoming): record the ids
    # placed anywhere beyond the first group as the "selected" set.
    response = payload.get("groupResp") or {}
    groups = response.get("groups") or []
    if not groups:
        return None
    indices = []
    for group in groups[1:]:
        for selected_id in group.get("ids") or []:
            found = _find_option(prompt, lambda option:
                                 option["payload"].get("id") == selected_id)
            if found is None:
                return None
            indices.append(found)
    return indices


def _match_auto_tap(prompt, payload):
    found = _find_option(prompt, lambda option: option["type"] == "AUTO_TAP")
    return [found] if found is not None else []


def _match_effect_cost(prompt, payload):
    response = payload.get("effectCostResp") or {}
    selection = response.get("costSelection") or {}
    indices = []
    for selected_id in selection.get("ids") or []:
        found = _find_option(prompt, lambda option: (
            option["payload"].get("instanceId") == selected_id
            or option["payload"].get("id") == selected_id))
        if found is None:
            return None
        indices.append(found)
    return indices


def _match_order(prompt, payload):
    response = payload.get("orderResp") or {}
    indices = []
    for selected_id in response.get("ids") or []:
        found = _find_option(prompt, lambda option:
                             option["payload"].get("id") == selected_id)
        if found is None:
            return None
        indices.append(found)
    return indices  # response order IS the chosen order


def _match_optional_action(prompt, payload):
    response = (payload.get("optionalResp") or {}).get("response") or ""
    if response.endswith("_No"):
        answer = "no"
    elif response.endswith("_Yes"):
        answer = "yes"
    else:
        return None
    found = _find_option(prompt, lambda option:
                         option["payload"].get("answer") == answer)
    return None if found is None else [found]


def _match_assign_damage(prompt, payload):
    response = payload.get("assignDamageResp") or {}
    indices = []
    for assigner in response.get("assigners") or []:
        assigner_id = assigner.get("instanceId")
        for assignment in assigner.get("assignments") or []:
            if not assignment.get("assignedDamage"):
                continue
            found = _find_option(prompt, lambda option: (
                option["payload"].get("assignerInstanceId") == assigner_id
                and option["payload"].get("instanceId")
                == assignment.get("instanceId")))
            if found is None:
                return None
            indices.append(found)
    return indices


_RESPONSE_MATCHERS = {
    "PerformActionResp": _match_perform_action,
    "ChooseStartingPlayerResp": _match_choose_starting_player,
    "MulliganResp": _match_mulligan,
    "SelectTargetsResp": _match_select_targets,
    "SubmitTargetsReq": _match_select_targets,
    "DeclareAttackersResp": _match_declare_attackers,
    "SubmitAttackersReq": _match_declare_attackers,
    "DeclareBlockersResp": _match_declare_blockers,
    "SubmitBlockersReq": _match_declare_blockers,
    "CastingTimeOptionsResp": _match_casting_time_options,
    "SearchResp": _match_search,
    "SelectNResp": _match_select_n,
    "GroupResp": _match_group,
    "PerformAutoTapActionsResp": _match_auto_tap,
    "EffectCostResp": _match_effect_cost,
    "OrderResp": _match_order,
    "OptionalActionResp": _match_optional_action,
    "AssignDamageResp": _match_assign_damage,
}


def _find_option(prompt, predicate):
    for option in _options(prompt):
        if predicate(option):
            return option["index"]
    return None


def _recipient_label(recipient):
    if recipient.get("type") == "DamageRecType_Player":
        return "player seat %s" % recipient.get("playerSystemSeatId")
    if "targetInstanceId" in recipient:
        return "instance %s" % recipient.get("targetInstanceId")
    return str(recipient or "target")


def _concede_prompt(event):
    return {
        "messageType": "ClientMessageType_ConcedeReq",
        "kind": "Concede",
        "msgId": None,
        "gameStateId": None,
        "timestamp": event.get("timestamp"),
        "select": {
            "type": "CONCEDE",
            "minCount": 1,
            "maxCount": 1,
            "option": [{"index": 0, "type": "CONCEDE", "label": "concede",
                        "payload": (event.get("payload") or {}).get("concedeReq") or {}}],
        },
        "snapshot": None,
    }
