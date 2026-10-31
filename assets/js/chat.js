// Chat functionality
// Handles user input, displays messages, and simulates bot response with typing animation.

document.addEventListener('DOMContentLoaded', () => {
  const userInput = document.getElementById('userInput');
  const sendBtn = document.getElementById('sendBtn');
  const messagesContainer = document.getElementById('messages');

  // Adjust textarea height automatically
  userInput.addEventListener('input', autoResize);
  function autoResize() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
  }

  // Send message on button click or Enter key (without Shift)
  sendBtn.addEventListener('click', sendMessage);
  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;
    appendMessage('user', text);
    userInput.value = '';
    autoResize.call(userInput);
    // Pass the user message to the bot response function
    simulateBotResponse(text);
  }

  function appendMessage(sender, text) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', sender);
    msgDiv.textContent = text;
    messagesContainer.appendChild(msgDiv);
    // Scroll to bottom smoothly
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // Fetch real answer from backend API
  async function simulateBotResponse(userMessage) {
    // Show typing indicator
    const typingDiv = document.createElement('div');
    typingDiv.classList.add('message', 'bot', 'typing');
    typingDiv.innerHTML = '<span></span><span></span><span></span>';
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    try {
      // Call backend API (replace URL with your actual endpoint)
      const responseText = await fetchChatResponse(userMessage);
      // Remove typing indicator and show real response
      messagesContainer.removeChild(typingDiv);
      appendMessage('bot', responseText);
    } catch (err) {
      // On error, fallback to placeholder and log
      console.error('Chat API error:', err);
      messagesContainer.removeChild(typingDiv);
      const fallback = "Sorry, I couldn't fetch an answer right now. Please try again later.";
      appendMessage('bot', fallback);
    }
  }

// Helper to call OpenAI API for real world knowledge answers.
async function fetchChatResponse(userMessage) {
  const apiUrl = 'https://api.openai.com/v1/chat/completions';
  const apiKey = 'YOUR_OPENAI_API_KEY'; // <-- replace with your actual API key
  const payload = {
    model: 'gpt-3.5-turbo',
    messages: [
      { role: 'system', content: 'You are a knowledgeable AI assistant with comprehensive world knowledge. Provide thorough, accurate, factual answers to any question.' },
      { role: 'user', content: userMessage }
    ],
    max_tokens: 500,
    temperature: 0.3
  };
  const res = await fetch(apiUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }
  const data = await res.json();
  // Extract the assistant's reply
  const reply = data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
  return reply ? reply.trim() : 'No response from AI';
}


  // No longer needed placeholder generator – kept for backward compatibility if desired.
  function generatePlaceholderResponse() {
    const responses = [
      "I'm here to help! What would you like to know?",
      "Sure, let me look that up for you.",
      "That's an interesting question. Let's explore it together.",
      "Absolutely! How can I assist you further?"
    ];
    return responses[Math.floor(Math.random() * responses.length)];
  }
});
