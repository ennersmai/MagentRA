<script setup>
import { ref, onMounted, nextTick } from 'vue'
import { useChatStore } from '@/stores/chat'

const store = useChatStore()
const message = ref('')
const chatContainer = ref(null)

const sendMessage = async () => {
  if (!message.value.trim()) return
  await store.sendMessage({
    text: message.value,
    sessionId: store.sessionId
  })
  message.value = ''
  await nextTick()
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
  }
}

onMounted(() => {
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
  }
})
</script>

<template>
  <div class="chat-wrapper">
    <header class="chat-header">
      <h2>I'm Magentra ask me anything, I'm here to help you!</h2>
    </header>
    <div class="chat-container" ref="chatContainer">
      <div

        v-for="msg in store.messages"
        :key="msg.id"
        :class="['message-bubble', msg.role]"
      >
        <div class="message-content">
          {{ msg.content }}
        </div>
      </div>
    </div>
    <form class="chat-input" @submit.prevent="sendMessage">
      <input
        v-model="message"
        placeholder="Type your message..."
        autocomplete="off"
      />
      <button type="submit">Send</button>
    </form>
  </div>
</template>

<style scoped>
.chat-wrapper {
  width: 100%;
  height: 100%;
  margin: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background-color: #f9f9f9;
}

.chat-header {
  background-color: #4f46e5;
  color: white;
  padding: 1rem;
  text-align: center;
}

.chat-container {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  background-color: #e5e7eb;
}

.message-bubble {
  margin-bottom: 1rem;
  max-width: 80%;
  padding: 0.75rem 1rem;
  border-radius: 16px;
  word-wrap: break-word;
}

.message-bubble.user {
  background-color: #4f46e5;
  color: #fff;
  align-self: flex-end;
  border-bottom-right-radius: 0;
}

.message-bubble.ai {
  background-color: #fff;
  color: #333;
  align-self: flex-start;
  border: 1px solid #ccc;
  border-bottom-left-radius: 0;
}

.chat-input {
  display: flex;
  padding: 1rem;
  border-top: 1px solid #ddd;
  background-color: #fff;
}

.chat-input input {
  flex: 1;
  padding: 0.75rem;
  font-size: 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-right: 0.5rem;
}

.chat-input button {
  padding: 0.75rem 1rem;
  font-size: 1rem;
  background-color: #4f46e5;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.chat-input button:hover {
  background-color: #4338ca;
}
</style> 