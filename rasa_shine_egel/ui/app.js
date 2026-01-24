const form = document.getElementById("composer");
const input = document.getElementById("composer-input");
const messages = document.getElementById("messages");
const statusPill = document.getElementById("status-pill");
const sendButton = form.querySelector(".send");
const navItems = document.querySelectorAll(".nav-item");
const quickButtons = document.querySelectorAll(".pill");

const senderIdKey = "rasa_sender_id";
const senderId =
  localStorage.getItem(senderIdKey) || `desktop-${crypto.randomUUID()}`;
localStorage.setItem(senderIdKey, senderId);

const endpoint = form.dataset.endpoint;
const baseUrl = endpoint.replace(/\/webhooks\/rest\/webhook\/?$/, "");

const addMessage = (text, role = "user") => {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  const meta = document.createElement("div");
  meta.className = "meta";
  const time = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
  meta.textContent = `${role === "user" ? "You" : "Assistant"} · ${time}`;

  wrapper.appendChild(bubble);
  wrapper.appendChild(meta);
  messages.appendChild(wrapper);
  messages.scrollTop = messages.scrollHeight;
};

const addImageMessage = (url, role = "assistant") => {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const img = document.createElement("img");
  img.src = url;
  img.alt = "image";
  img.style.maxWidth = "260px";
  img.style.borderRadius = "12px";
  bubble.appendChild(img);

  const meta = document.createElement("div");
  meta.className = "meta";
  const time = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
  meta.textContent = `Assistant · ${time}`;

  wrapper.appendChild(bubble);
  wrapper.appendChild(meta);
  messages.appendChild(wrapper);
  messages.scrollTop = messages.scrollHeight;
};

const setStatus = (text, connected) => {
  statusPill.textContent = text;
  statusPill.style.borderColor = connected
    ? "rgba(42, 215, 255, 0.6)"
    : "rgba(255, 189, 107, 0.6)";
  statusPill.style.color = connected ? "var(--accent-2)" : "#ffbd6b";
};

const sendToRasa = async (message) => {
  const payload = { sender: senderId, message };
  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }
  return res.json();
};

const sendMessage = async (text) => {
  addMessage(text, "user");
  sendButton.disabled = true;
  sendButton.textContent = "Sending...";

  try {
    const replies = await sendToRasa(text);
    setStatus("Connected", true);
    if (!replies.length) {
      addMessage(
        "Хариу ирсэнгүй. Action server log-оо шалгаарай.",
        "assistant"
      );
      return;
    }
    replies.forEach((reply) => {
      if (reply.text) {
        addMessage(reply.text, "assistant");
      } else if (reply.image) {
        addImageMessage(reply.image, "assistant");
      } else if (reply.custom) {
        addMessage(JSON.stringify(reply.custom), "assistant");
      }
    });
  } catch (err) {
    setStatus("Disconnected", false);
    addMessage(
      "Холболт амжилтгүй. Rasa сервер ажиллаж байгаа эсэхийг шалгана уу.",
      "assistant"
    );
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "Send";
  }
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  await sendMessage(text);
});

quickButtons.forEach((btn) => {
  btn.addEventListener("click", (event) => {
    event.preventDefault();
    const text = btn.dataset.message || btn.textContent.trim();
    if (!text) return;
    sendMessage(text);
  });
});

navItems.forEach((item) => {
  item.addEventListener("click", () => {
    navItems.forEach((n) => n.classList.remove("active"));
    item.classList.add("active");
  });
});

const checkStatus = async () => {
  try {
    const res = await fetch(`${baseUrl}/status`);
    setStatus(res.ok ? "Connected" : "Disconnected", res.ok);
  } catch {
    setStatus("Disconnected", false);
  }
};

checkStatus();
