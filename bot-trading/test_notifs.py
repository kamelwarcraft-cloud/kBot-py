import os, sys, json, datetime as dt, requests
from dotenv import load_dotenv

load_dotenv()

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID")
DC_HOOK  = os.getenv("DISCORD_WEBHOOK_URL")

def send_telegram(msg: str):
    if not TG_TOKEN:
        print("🔸 Telegram ignoré (TELEGRAM_BOT_TOKEN manquant dans .env)")
        return
    if not TG_CHAT:
        print("❗ TELEGRAM_CHAT_ID manquant. Tentative de découverte via getUpdates…")
        try:
            r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates", timeout=10)
            data = r.json()
            chats = []
            for upd in data.get("result", []):
                for key in ("message","edited_message","channel_post","edited_channel_post","callback_query"):
                    if key in upd:
                        obj = upd[key]
                        if key == "callback_query":
                            obj = upd[key].get("message", {})
                        chat = obj.get("chat", {})
                        cid = chat.get("id")
                        title = chat.get("title") or chat.get("username") or chat.get("first_name")
                        if cid:
                            chats.append((cid, title))
            uniq = []
            seen = set()
            for cid, title in chats:
                if cid not in seen:
                    uniq.append((cid, title))
                    seen.add(cid)
            if not uniq:
                print("   → Aucun chat trouvé. Envoie un message à ton bot puis relance.")
            else:
                print("   → Chats détectés (mets-en un dans TELEGRAM_CHAT_ID) :")
                for cid, title in uniq:
                    print(f"     - chat_id={cid}  ({title or 'sans titre'})")
        except Exception as e:
            print(f"   → Erreur getUpdates: {e}")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg, "parse_mode":"HTML"},
            timeout=10
        )
        r.raise_for_status()
        print("✅ Telegram: message envoyé.")
    except Exception as e:
        print(f"❌ Telegram: échec d'envoi → {e}")

def send_discord(msg: str):
    if not DC_HOOK:
        print("🔸 Discord ignoré (DISCORD_WEBHOOK_URL manquant dans .env)")
        return
    try:
        r = requests.post(DC_HOOK, json={"content": msg}, timeout=10)
        r.raise_for_status()
        print("✅ Discord: message envoyé.")
    except Exception as e:
        print(f"❌ Discord: échec d'envoi → {e}")

if __name__ == "__main__":
    custom = " ".join(sys.argv[1:]).strip()
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = custom if custom else f"✅ Test notifications OK — {timestamp}"
    print("➤ Envoi des notifications…")
    send_telegram(msg)
    send_discord(msg)
    print("➤ Terminé.")
