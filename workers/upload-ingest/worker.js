const MAX_BODY_BYTES = 5 * 1024 * 1024;
const MAX_RECORDS = 20000;
const REQUIRED_KIND = "magic_cabt_upload_bundle";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/health") {
      return json({ ok: true, service: "magic-cabt-upload-ingest" });
    }
    if (request.method !== "POST" || url.pathname !== "/bundles") {
      return json({ ok: false, error: "NOT_FOUND" }, 404);
    }
    const authError = checkAuth(request, env);
    if (authError) return authError;
    const length = Number(request.headers.get("content-length") || "0");
    if (length <= 0 || length > MAX_BODY_BYTES) {
      return json({ ok: false, error: "BAD_SIZE" }, 413);
    }
    let envelope;
    try {
      envelope = await request.json();
    } catch (error) {
      return json({ ok: false, error: "BAD_JSON" }, 400);
    }
    const validation = validateEnvelope(envelope);
    if (!validation.ok) return json(validation, 400);

    const bundleId = await stableBundleId(envelope);
    const stored = {
      bundleId,
      receivedAt: new Date().toISOString(),
      remoteCountry: request.cf && request.cf.country,
      envelope,
    };
    if (env.BUNDLES) {
      await env.BUNDLES.put(`bundles/${bundleId}.json`, JSON.stringify(stored));
    }
    return json({ ok: true, bundleId, acceptedRecords: envelope.decisions.length });
  },
};

function checkAuth(request, env) {
  if (!env.UPLOAD_TOKEN) return null;
  const expected = `Bearer ${env.UPLOAD_TOKEN}`;
  if (request.headers.get("authorization") !== expected) {
    return json({ ok: false, error: "UNAUTHORIZED" }, 401);
  }
  return null;
}

function validateEnvelope(envelope) {
  if (!envelope || typeof envelope !== "object") return fail("BAD_ENVELOPE");
  if (envelope.kind !== REQUIRED_KIND) return fail("BAD_KIND");
  if (!envelope.consent || envelope.consent.allowTraining !== true) return fail("NO_TRAINING_CONSENT");
  if (!Array.isArray(envelope.decisions)) return fail("NO_DECISIONS");
  if (envelope.decisions.length < 1 || envelope.decisions.length > MAX_RECORDS) return fail("BAD_RECORD_COUNT");
  if (!envelope.manifest || typeof envelope.manifest !== "object") return fail("NO_MANIFEST");
  if (!envelope.redactionReport || envelope.redactionReport.rawPlayerLogUploaded !== false) {
    return fail("RAW_LOG_UPLOAD_REJECTED");
  }
  for (let i = 0; i < envelope.decisions.length; i += 1) {
    const decision = envelope.decisions[i];
    if (!decision || typeof decision !== "object") return fail("BAD_DECISION", i);
    if (!decision.observation && !decision.select) return fail("DECISION_NO_PROMPT", i);
    const selected = decision.selectedIndices || decision.selected;
    if (!Array.isArray(selected)) return fail("DECISION_NO_SELECTION", i);
  }
  return { ok: true };
}

function fail(error, index) {
  const payload = { ok: false, error };
  if (index !== undefined) payload.index = index;
  return payload;
}

async function stableBundleId(envelope) {
  const bytes = new TextEncoder().encode(JSON.stringify({
    contributorId: envelope.contributorId || "anonymous",
    createdAtUnix: envelope.createdAtUnix,
    recordCount: envelope.decisions.length,
    manifest: envelope.manifest,
  }));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 32);
}

function json(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" },
  });
}
