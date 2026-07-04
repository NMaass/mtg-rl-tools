# CABT adapter verification

One command runs the whole adapter suite from the repo root:

```sh
scripts/run-cabt-adapter-tests.sh
```

CI can call that script, or the equivalent Maven/Python commands below.

## Test layers

### Java unit + callback-boundary integration tests

```sh
mvn -pl Mage.Server.Plugins/Mage.Player.AI test
```

The module's test tree (`src/test/java/mage/player/cabt/`) contains only the
CABT adapter tests. Two layers live there:

- **Unit**: prompt builders, selection appliers, validators, serializers,
  the card-data exporter, and the dataset writer, each against stub games
  (`StubGames`) carrying real engine containers (Battlefield, Combat,
  GameState, ManaPool, ...).
- **Callback-boundary integration** (`CabtBridgePlayer*Test`): the exact
  `Player` callback the engine invokes (chooseTarget, playMana,
  selectAttackers, chooseMulligan, ...) driven end to end through
  controller → validation → applier → trace, asserting on real engine state
  afterwards.

Full-engine smoke games (boot a game, play scripted turns over the
subprocess protocol) are Task 20 and do not exist yet; when delivered they
join the module suite and this script.

### Python unit tests

```sh
cd python && python3 -m unittest discover -s tests
```

Tests `magic_cabt` (card-data parsing, JSONL dataset reading) against
fixtures that are **real Java output**: `MagicCardDataExporterTest` and
`CabtDatasetWriterTest` regenerate them under
`Mage.Server.Plugins/Mage.Player.AI/target/cabt-fixtures/` on every Java
run, and the script copies them over `python/tests/fixtures/` before the
Python layer runs — so a Java-side format change fails the Python tests in
the same script run.

### End-to-end protocol smoke tests

Not available yet: they need the subprocess protocol server (Task 18) and
Python client (Task 19). The script auto-enables this layer when
`mage.player.cabt.CabtProtocolServer` appears in the module's compiled
classes, and runs `python/tests/test_protocol*.py` against it.

## When a test fails, look here

| Failure | Likely task area |
| --- | --- |
| `SelectionValidatorTest`, `InvalidSelectionException` in many tests | Selection plumbing (Tasks 1–2) |
| `MagicObservation*Test`, `MagicObjectViewFactoryTest` | Public state serialization / hidden-info boundary (Task 6) |
| `CabtTargetPromptBuilderTest`, `CabtBridgePlayerTargetPromptTest` | Target prompts (Task 7) |
| `CabtYesNoPromptTest`, `CabtChoicePromptTest`, `CabtPilePromptTest` | Generic choice prompts (Task 8) |
| `CabtModePromptBuilderTest`, `CabtBridgePlayerModePromptTest` | Mode selection (Task 9) |
| `CabtNumberPromptTest`, `CabtMultiAmountPromptTest` | Numeric/X/multi-amount prompts (Task 10) |
| `CabtTriggeredAbility*Test` | Trigger ordering (Task 11) |
| `CabtReplacementEffect*Test` | Replacement-effect choice (Task 12) |
| `CabtManaPromptBuilderTest`, `CabtBridgePlayerManaPromptTest` | Mana payment (Task 13) |
| `CabtAttackers*Test`, `CabtBlockers*Test` | Combat declarations (Tasks 14–15) |
| `CabtMulliganPromptTest`, `CabtBridgePlayerMulliganTest` | Mulligan (Task 16) |
| `MagicCardDataExporterTest`, `python test_card_data` | Static card data export (Task 21) |
| `CabtDatasetWriterTest`, `python test_dataset` | Dataset writer/reader (Task 22) |
| `CabtPromptAuditTest.allSurfacedPromptsHaveTests` | A surfaced prompt lost its implementation or test class — fix the audit entry or restore the class (Task 23) |
| `CabtPromptAuditTest.failClosedPromptsThrow...` | A FAIL_CLOSED callback stopped throwing `CabtUnhandledDecisionException` — a decision may be silently AI-decided (Task 23) |
| `CabtDecisionSurfaceAuditTest` | Audit drift: entry names/statuses no longer match the engine or bridge (Tasks 3, 23) |
| Python `test_card_data` passes locally but fails after the script's fixture refresh | The Java output format changed; update the Python parser or the Java shape together (Tasks 21–22) |

A prompt failing **in a real game** (a `CabtUnhandledDecisionException` in a
game log) means the engine hit a surface the bridge fails closed on — check
`CabtDecisionSurfaceAudit` for the callback's entry; its adapterPlan says
what is needed to surface it.

## Adding a new prompt surface

1. Add the audit entry in `CabtDecisionSurfaceAudit` (real engine signature,
   implementation class, test class).
2. Implement the builder/applier and the `CabtBridgePlayer` override.
3. Add the unit + callback-boundary tests named by the entry.

`CabtPromptAuditTest` fails until all three exist — that is the coverage
gate working as intended.
