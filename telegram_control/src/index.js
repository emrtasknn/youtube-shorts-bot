const JSON_HEADERS = { "content-type": "application/json; charset=utf-8" };

function json(data, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: JSON_HEADERS });
}

async function telegram(env, method, body) {
  const res = await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/${method}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Telegram API ${method} failed: ${res.status}`);
  return res.json();
}

async function githubDispatch(env, workflow, inputs) {
  const res = await fetch(
    `https://api.github.com/repos/${env.GITHUB_REPO}/actions/workflows/${workflow}/dispatches`,
    {
      method: "POST",
      headers: {
        "Accept": "application/vnd.github+json",
        "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
        "X-GitHub-Api-Version": "2026-03-10",
        "User-Agent": "youtube-shorts-control",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ref: "main", inputs }),
    }
  );
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`GitHub dispatch failed: ${res.status} ${detail}`);
  }
}

async function saveState(env, runId, state) {
  await env.APPROVALS.put(`approval:${runId}`, JSON.stringify(state));
}

async function getState(env, runId) {
  return env.APPROVALS.get(`approval:${runId}`, "json");
}

async function answerCallback(env, callbackId, text) {
  return telegram(env, "answerCallbackQuery", {
    callback_query_id: callbackId,
    text,
    show_alert: false,
  });
}

async function removeButtons(env, chatId, messageId) {
  return telegram(env, "editMessageReplyMarkup", {
    chat_id: chatId,
    message_id: messageId,
    reply_markup: { inline_keyboard: [] },
  });
}

async function handleCallback(env, callback) {
  const message = callback.message;
  if (!message || String(message.chat.id) !== String(env.TELEGRAM_CHAT_ID)) {
    return;
  }

  const [action, runId] = String(callback.data || "").split(":", 2);
  if (!action || !runId) return;

  const state = await getState(env, runId);
  if (!state) {
    await answerCallback(env, callback.id, "Bu video kaydı artık bulunamıyor.");
    return;
  }

  if (action === "script") {
    await answerCallback(env, callback.id, "Senaryo gönderiliyor...");
    const scenes = state.scenes || [];
    const lines = [`📜 ${state.topic?.title || "Short"} — Senaryo`, ""];
    scenes.forEach((scene, i) => {
      lines.push(`🎬 ${i + 1}. Sahne`);
      lines.push(scene.narration || "");
      lines.push("");
    });
    await telegram(env, "sendMessage", {
      chat_id: message.chat.id,
      text: lines.join("\n").slice(0, 4000),
    });
    return;
  }

  if (state.status !== "pending") {
    await answerCallback(env, callback.id, `Bu video zaten ${state.status} durumunda.`);
    return;
  }

  if (action === "cancel") {
    state.status = "cancelled";
    state.updated_at = new Date().toISOString();
    await saveState(env, runId, state);
    await answerCallback(env, callback.id, "Video iptal edildi.");
    await removeButtons(env, message.chat.id, message.message_id);
    await telegram(env, "sendMessage", {
      chat_id: message.chat.id,
      text: `❌ ${state.topic?.title || "Video"} iptal edildi.`,
    });
    return;
  }

  if (action === "publish") {
    state.status = "publishing";
    state.updated_at = new Date().toISOString();
    await saveState(env, runId, state);
    await answerCallback(env, callback.id, "Yayınlama başlatılıyor...");
    await removeButtons(env, message.chat.id, message.message_id);
    await telegram(env, "sendMessage", {
      chat_id: message.chat.id,
      text: `🚀 ${state.topic?.title || "Video"} YouTube'a yükleniyor...`,
    });

    try {
      await githubDispatch(env, "publish_short.yml", {
        source_run_id: String(state.github_run_id),
        source_run_number: String(state.github_run_number),
        approval_run_id: runId,
      });
    } catch (err) {
      state.status = "publish_dispatch_failed";
      state.error = String(err);
      await saveState(env, runId, state);
      await telegram(env, "sendMessage", {
        chat_id: message.chat.id,
        text: `⚠️ Yayınlama başlatılamadı.\\n${String(err).slice(0, 1000)}`,
      });
    }
    return;
  }

  if (action === "regen") {
    state.status = "regenerating";
    state.updated_at = new Date().toISOString();
    await saveState(env, runId, state);
    await answerCallback(env, callback.id, "Yeni üretim başlatılıyor...");
    await removeButtons(env, message.chat.id, message.message_id);
    await telegram(env, "sendMessage", {
      chat_id: message.chat.id,
      text: `🔄 ${state.topic?.title || "Video"} reddedildi. Yeni bir tarih olayı aranıyor...`,
    });

    try {
      await githubDispatch(env, "daily_short.yml", {
        mode: "regenerate",
        source_run_id: String(state.github_run_id),
        source_event_id: String(state.event_id || ""),
      });
    } catch (err) {
      state.status = "regen_dispatch_failed";
      state.error = String(err);
      await saveState(env, runId, state);
      await telegram(env, "sendMessage", {
        chat_id: message.chat.id,
        text: `⚠️ Yeniden üretim başlatılamadı.\\n${String(err).slice(0, 1000)}`,
      });
    }
    return;
  }

  await answerCallback(env, callback.id, "Bilinmeyen işlem.");
}

export default {
  async fetch(request, env) {
    try {
      const url = new URL(request.url);

      if (request.method === "GET" && url.pathname === "/health") {
        return json({ ok: true, service: "telegram-control" });
      }

      if (request.method === "GET" && url.pathname === "/github-check") {
        const checkUrl = `https://api.github.com/repos/${env.GITHUB_REPO}/actions/workflows/publish_short.yml`;
        const res = await fetch(checkUrl, {
          headers: {
            "Accept": "application/vnd.github+json",
            "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
            "X-GitHub-Api-Version": "2026-03-10",
            "User-Agent": "youtube-shorts-control",
          },
        });

        if (res.ok) {
          const data = await res.json();
          return json({
            ok: true,
            github_status: res.status,
            repository: env.GITHUB_REPO,
            workflow: data.path || "publish_short.yml",
            state: data.state || null,
          });
        }

        return json({
          ok: false,
          github_status: res.status,
          repository: env.GITHUB_REPO,
          reason: res.status === 401
            ? "GitHub token is invalid or expired."
            : res.status === 403
              ? "GitHub token is valid but lacks required permission."
              : res.status === 404
                ? "Repository/workflow is not accessible with this token."
                : "GitHub API request failed.",
        }, 502);
      }

      if (request.method !== "POST") return json({ error: "method_not_allowed" }, 405);

      const secret = request.headers.get("X-Control-Secret");
      const telegramSecret = request.headers.get("X-Telegram-Bot-Api-Secret-Token");

      if (url.pathname === "/register") {
        if (!secret || secret !== env.CONTROL_API_SECRET) return json({ error: "unauthorized" }, 401);
        const state = await request.json();
        if (!state?.run_id) return json({ error: "run_id_required" }, 400);
        await saveState(env, state.run_id, state);
        return json({ ok: true });
      }

      if (url.pathname === "/telegram") {
        if (!telegramSecret || telegramSecret !== env.TELEGRAM_WEBHOOK_SECRET) {
          return json({ error: "unauthorized" }, 401);
        }
        const update = await request.json();
        if (update.callback_query) await handleCallback(env, update.callback_query);
        return json({ ok: true });
      }

      if (url.pathname === "/publish-result") {
        if (!secret || secret !== env.CONTROL_API_SECRET) return json({ error: "unauthorized" }, 401);
        const result = await request.json();
        if (!result?.run_id) return json({ error: "run_id_required" }, 400);
        const state = await getState(env, result.run_id);
        if (!state) return json({ error: "approval_not_found" }, 404);

        state.status = result.status || "published";
        state.youtube_url = result.youtube_url || "";
        state.youtube_video_id = result.youtube_video_id || "";
        state.error = result.error || "";
        state.updated_at = new Date().toISOString();
        await saveState(env, result.run_id, state);

        const title = state.topic?.title || "Video";
        if (state.status === "published") {
          await telegram(env, "sendMessage", {
            chat_id: env.TELEGRAM_CHAT_ID,
            text: `🎉 ${title} başarıyla YouTube Shorts'a yüklendi.\\n\\n🔗 ${state.youtube_url}`,
          });
        } else {
          await telegram(env, "sendMessage", {
            chat_id: env.TELEGRAM_CHAT_ID,
            text: `⚠️ ${title} YouTube'a yüklenemedi.\\n\\n${String(state.error || "Bilinmeyen hata").slice(0, 1500)}`,
          });
        }
        return json({ ok: true });
      }

      return json({ error: "not_found" }, 404);
    } catch (err) {
      console.error(err);
      return json({ error: String(err) }, 500);
    }
  },
};
