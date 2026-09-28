const JSON_HEADERS = { "content-type": "application/json; charset=utf-8" };

function json(data, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: JSON_HEADERS });
}

async function tiktokToken(env, grantType, extra = {}) {
  const body = new URLSearchParams({
    client_key: env.TIKTOK_CLIENT_KEY,
    client_secret: env.TIKTOK_CLIENT_SECRET,
    grant_type: grantType,
    ...extra,
  });
  const res = await fetch("https://open.tiktokapis.com/v2/oauth/token/", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  const data = await res.json();
  if (!res.ok || data.error) {
    throw new Error(`TikTok OAuth failed: ${res.status} ${JSON.stringify(data)}`);
  }
  return data;
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

function legalPage(title, body) {
  return new Response(`<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>${title} — Bir Garip Tarih</title>
  <style>
    body{font-family:Arial,Helvetica,sans-serif;max-width:820px;margin:0 auto;padding:40px 22px;line-height:1.65;color:#222;background:#faf9f7}
    h1{line-height:1.2} h2{margin-top:28px} a{color:#6b4b2a}
    .card{background:#fff;border:1px solid #e5e1dc;border-radius:14px;padding:28px;box-shadow:0 4px 18px rgba(0,0,0,.04)}
    footer{margin-top:32px;color:#666;font-size:14px}
  </style>
</head>
<body><main class="card"><h1>${title}</h1>${body}
<footer>Bir Garip Tarih · <a href="/terms">Terms of Service</a> · <a href="/privacy">Privacy Policy</a></footer>
</main></body></html>`, {status:200, headers:{"content-type":"text/html; charset=utf-8"}});
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
    const lines = [`📜 ${state.topic?.youtube_title || state.topic?.title || "Short"} — Senaryo`, ""];
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
      text: `❌ ${state.topic?.youtube_title || state.topic?.youtube_title || state.topic?.title || "Video"} iptal edildi.`,
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
      text: `🚀 ${state.topic?.youtube_title || state.topic?.youtube_title || state.topic?.title || "Video"} YouTube'a yükleniyor...`,
    });

    try {
      await githubDispatch(env, "publish_short.yml", {
        source_run_id: String(state.github_run_id),
        source_run_number: String(state.github_run_number),
        approval_run_id: runId,
        youtube_title: String(state.topic?.youtube_title || state.topic?.title || "Tarihin Bilinmeyen Gizemi").slice(0, 90),
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
      text: `🔄 ${state.topic?.youtube_title || state.topic?.youtube_title || state.topic?.title || "Video"} reddedildi. Yeni bir tarih olayı aranıyor...`,
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

async function handleMessage(env, message) {
  if (!message || String(message.chat?.id) !== String(env.TELEGRAM_CHAT_ID)) return;
  const text = String(message.text || "").trim();
  if (!text.startsWith("/")) return;

  const match = text.match(/^\/(custom|ayt|today|generate)(?:@\w+)?(?:\s+([\\s\\S]*))?$/i);
  if (!match) return;

  const command = match[1].toLowerCase();
  const argument = String(match[2] || "").trim();
  let contentType = "TREND_HISTORY";
  let customPrompt = "";

  if (command === "custom") {
    contentType = "CUSTOM";
    customPrompt = argument;
    if (!customPrompt) {
      await telegram(env, "sendMessage", {
        chat_id: message.chat.id,
        text: "🧪 Kullanım: /custom [üretmek istediğin tarih videosu fikri]",
      });
      return;
    }
  } else if (command === "ayt") {
    contentType = "AYT_HISTORY";
  } else if (command === "today") {
    contentType = "TODAY_IN_HISTORY";
  }

  try {
    await githubDispatch(env, "daily_short.yml", {
      mode: command === "custom" ? "custom" : command,
      batch_count: "1",
      content_type: contentType,
      custom_prompt: customPrompt,
    });
    const labels = {
      TREND_HISTORY: "Trend History",
      TODAY_IN_HISTORY: "Tarihte Bugün",
      AYT_HISTORY: "AYT Tarih",
      CUSTOM: "Custom",
    };
    await telegram(env, "sendMessage", {
      chat_id: message.chat.id,
      text: `🚀 ${labels[contentType] || contentType} üretimi başlatıldı. Tamamlandığında önizleme Telegram'a gelecek.`,
    });
  } catch (err) {
    await telegram(env, "sendMessage", {
      chat_id: message.chat.id,
      text: `⚠️ Üretim başlatılamadı.\n${String(err).slice(0, 1000)}`,
    });
  }
}

export default {
  async fetch(request, env) {
    try {
      const url = new URL(request.url);

      if (request.method === "GET" && url.pathname === "/") {
        return legalPage("Bir Garip Tarih", `
<p>Bir Garip Tarih, kısa ve ilgi çekici tarih videoları hazırlayan bir içerik platformudur.</p>
<p>Bu web alanı, içerik yayınlama entegrasyonları ve uygulama bilgileri için kullanılmaktadır.</p>
`);
      }

      if (
        (request.method === "GET" || request.method === "HEAD") &&
        url.pathname === "/tiktok-developers-site-verification.txt"
      ) {
        return new Response("tiktok-developers-site-verification=1uhsfoXkPy9abNfuJa6DFFOYdCVndMN8", {
          status: 200,
          headers: {
            "content-type": "text/plain; charset=utf-8",
            "cache-control": "no-store",
          },
        });
      }

      if (
        (request.method === "GET" || request.method === "HEAD") &&
        url.pathname === "/tiktok-developers-site-verification=1uhsfoXkPy9abNfuJa6DFFOYdCVndMN8"
      ) {
        return new Response("tiktok-developers-site-verification=1uhsfoXkPy9abNfuJa6DFFOYdCVndMN8", {
          status: 200,
          headers: {
            "content-type": "text/plain; charset=utf-8",
            "cache-control": "no-store",
          },
        });
      }

      if (request.method === "GET" && url.pathname === "/terms") {
        return legalPage("Terms of Service", `
<p><strong>Son güncelleme:</strong> 28 Eylül 2026</p>
<h2>1. Hizmet</h2>
<p>Bir Garip Tarih, kısa tarih içeriklerinin hazırlanması, yönetilmesi ve yetkilendirilmiş sosyal medya hesaplarında paylaşılması için kullanılan bir içerik platformudur.</p>
<h2>2. Kullanıcı Yetkilendirmesi</h2>
<p>Sosyal medya hesaplarına içerik gönderme özellikleri yalnızca hesap sahibi tarafından açıkça yetkilendirildiğinde kullanılır. Kullanıcı, yetkilendirmeyi istediği zaman ilgili platformun hesap ve uygulama ayarlarından kaldırabilir.</p>
<h2>3. İçerik Yayınlama</h2>
<p>Bir Garip Tarih, yetkilendirilmiş hesabın onayı olmadan içerik yayınlamayı amaçlamaz. Kullanıcı tarafından onaylanan içerikler ilgili platformun API kurallarına uygun şekilde gönderilir.</p>
<h2>4. Sorumluluk</h2>
<p>Kullanıcı, hesabı üzerinden yayınlanan içeriklerin ve hesabın kullanımının ilgili platformların kurallarına uygun olmasından sorumludur. Hizmet, üçüncü taraf platformların kesintilerinden veya politika değişikliklerinden etkilenebilir.</p>
<h2>5. İletişim</h2>
<p>İletişim: birgariptarihofficial@gmail.com</p>
`);
      }

      if (request.method === "GET" && url.pathname === "/privacy") {
        return legalPage("Privacy Policy", `
<p><strong>Son güncelleme:</strong> 28 Eylül 2026</p>
<h2>1. Toplanan Bilgiler</h2>
<p>Uygulama, sosyal medya hesabı yetkilendirmesi sırasında ilgili platform tarafından sağlanan hesap kimliği, yetkilendirme kapsamı ve erişim yenileme bilgileri gibi teknik verileri, entegrasyonu çalıştırmak için gerekli olduğu ölçüde saklayabilir.</p>
<h2>2. TikTok Verileri</h2>
<p>TikTok entegrasyonunda erişim ve yenileme belirteçleri yalnızca yetkilendirilmiş hesabın içerik yayınlama işlemlerini gerçekleştirmek ve erişimi yenilemek amacıyla kullanılır. Bu bilgiler herkese açık olarak paylaşılmaz.</p>
<h2>3. Kullanım Amacı</h2>
<p>Veriler; hesap yetkilendirmesini yönetmek, onaylanan videoları yayınlamak, yayınlama durumunu takip etmek ve teknik sorunları gidermek amacıyla kullanılır.</p>
<h2>4. Saklama ve Güvenlik</h2>
<p>Yetkilendirme verileri erişimi kısıtlanmış uygulama altyapısında saklanır. Kullanıcı yetkilendirmeyi kaldırdığında veya entegrasyon artık gerekli olmadığında ilgili kimlik doğrulama verileri silinebilir.</p>
<h2>5. Üçüncü Taraf Hizmetler</h2>
<p>İçerik yayınlama özellikleri TikTok gibi üçüncü taraf platformların API'lerini kullanabilir. Bu platformların kendi gizlilik politikaları ve kullanım koşulları ayrıca geçerlidir.</p>
<h2>6. İletişim</h2>
<p>Gizlilikle ilgili sorular için: birgariptarihofficial@gmail.com</p>
`);
      }

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

      if (request.method === "GET" && url.pathname === "/tiktok/connect") {
        if (!env.TIKTOK_CLIENT_KEY || !env.TIKTOK_REDIRECT_URI) {
          return json({ error: "TikTok OAuth is not configured." }, 500);
        }
        const state = crypto.randomUUID();
        await env.APPROVALS.put(
          `tiktok:oauth-state:${state}`,
          JSON.stringify({ created_at: new Date().toISOString() }),
          { expirationTtl: 600 }
        );
        const params = new URLSearchParams({
          client_key: env.TIKTOK_CLIENT_KEY,
          response_type: "code",
          scope: "video.publish",
          redirect_uri: env.TIKTOK_REDIRECT_URI,
          state,
        });
        return Response.redirect(
          `https://www.tiktok.com/v2/auth/authorize/?${params.toString()}`,
          302
        );
      }

      if (request.method === "GET" && url.pathname === "/tiktok/callback") {
        const state = url.searchParams.get("state") || "";
        const code = url.searchParams.get("code") || "";
        const error = url.searchParams.get("error") || "";
        const errorDescription = url.searchParams.get("error_description") || "";
        if (error) return new Response(`TikTok authorization failed: ${error} ${errorDescription}`, { status: 400 });
        if (!state || !code) return new Response("Missing TikTok OAuth state/code.", { status: 400 });

        const stateKey = `tiktok:oauth-state:${state}`;
        const stateRecord = await env.APPROVALS.get(stateKey, "json");
        if (!stateRecord) return new Response("TikTok OAuth state expired or invalid.", { status: 400 });
        await env.APPROVALS.delete(stateKey);

        const token = await tiktokToken(env, "authorization_code", {
          code,
          redirect_uri: env.TIKTOK_REDIRECT_URI,
        });
        if (!token.refresh_token) return new Response("TikTok did not return a refresh token.", { status: 502 });

        await env.APPROVALS.put("tiktok:credentials", JSON.stringify({
          refresh_token: token.refresh_token,
          open_id: token.open_id || "",
          scope: token.scope || "",
          updated_at: new Date().toISOString(),
        }));

        return new Response(
          "TikTok connected successfully. You can close this tab and return to Telegram.",
          { status: 200, headers: { "content-type": "text/plain; charset=utf-8" } }
        );
      }

      if (request.method === "POST" && url.pathname === "/tiktok/access-token") {
        const secret = request.headers.get("X-Control-Secret");
        if (!secret || secret !== env.CONTROL_API_SECRET) return json({ error: "unauthorized" }, 401);
        if (!env.TIKTOK_CLIENT_KEY || !env.TIKTOK_CLIENT_SECRET) {
          return json({ error: "TikTok OAuth is not configured." }, 500);
        }

        const credentials = await env.APPROVALS.get("tiktok:credentials", "json");
        if (!credentials?.refresh_token) {
          return json({ error: "TikTok account is not connected. Open /tiktok/connect first." }, 409);
        }

        const token = await tiktokToken(env, "refresh_token", {
          refresh_token: credentials.refresh_token,
        });
        await env.APPROVALS.put("tiktok:credentials", JSON.stringify({
          refresh_token: token.refresh_token || credentials.refresh_token,
          open_id: token.open_id || credentials.open_id || "",
          scope: token.scope || credentials.scope || "",
          updated_at: new Date().toISOString(),
        }));
        return json({
          access_token: token.access_token,
          expires_in: token.expires_in || 0,
          open_id: token.open_id || credentials.open_id || "",
        });
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
        if (update.message) await handleMessage(env, update.message);
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
        state.playlist = result.playlist || {};
        state.tiktok_url = result.tiktok_url || "";
        state.tiktok_publish_id = result.tiktok_publish_id || "";
        state.tiktok_post_id = result.tiktok_post_id || "";
        state.tiktok_status = result.tiktok_status || "";
        state.tiktok_fail_reason = result.tiktok_fail_reason || "";
        state.tiktok_error = result.tiktok_error || "";
        state.error = result.error || "";
        state.updated_at = new Date().toISOString();
        await saveState(env, result.run_id, state);

        const title = state.topic?.youtube_title || state.topic?.youtube_title || state.topic?.title || "Video";
        if (state.status === "published") {
          const tiktokLine = state.tiktok_url
            ? `\\n\\n🎵 TikTok: ${state.tiktok_url}`
            : state.tiktok_status
              ? `\\n\\n🎵 TikTok: ${state.tiktok_status}${state.tiktok_fail_reason ? ` — ${state.tiktok_fail_reason}` : ""}`
              : state.tiktok_error
                ? `\\n\\n⚠️ TikTok yüklemesi başarısız: ${state.tiktok_error}`
                : "";
          await telegram(env, "sendMessage", {
            chat_id: env.TELEGRAM_CHAT_ID,
            text: `🎉 ${title} yayınlandı.\\n\\n▶️ YouTube: ${state.youtube_url}${tiktokLine}${state.playlist?.status === "added" ? "\\n📚 Playlist'e eklendi." : state.playlist?.status === "failed" ? "\\n⚠️ Playlist'e eklenemedi; YouTube videosu yayında." : ""}`,
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
