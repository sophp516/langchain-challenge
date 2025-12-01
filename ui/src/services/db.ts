// API functions for MongoDB operations via backend server
// The backend server handles the actual MongoDB connection

const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:3001'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

export interface ThreadData {
  thread_id: string
  messages: Message[]
  created_at: Date
  updated_at: Date
}

/**
 * Create a new thread document in MongoDB
 * Called when user starts a new conversation
 */
export async function createThread(threadId: string): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/threads`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ thread_id: threadId }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`Failed to create thread: ${response.status} - ${errorText}`)
    }

    console.log(`Created new thread: ${threadId}`)
  } catch (error) {
    console.error('Error creating thread:', error)
    throw error
  }
}

/**
 * Append a message to an existing thread
 * Called each time a new message is added to the conversation
 */
export async function appendMessage(
  threadId: string,
  message: Message
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/threads/${threadId}/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: [{
          id: message.id,
          role: message.role,
          content: message.content,
          timestamp: message.timestamp
        }]
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`Failed to append message: ${response.status} - ${errorText}`)
    }

    console.log(`Appended message to thread ${threadId}`)
  } catch (error) {
    console.error('Error appending message:', error)
    throw error
  }
}

/**
 * Save messages for a thread to MongoDB
 */
export async function saveThreadMessages(
  threadId: string,
  messages: Message[]
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/threads/${threadId}/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: messages.map(msg => ({
          id: msg.id,
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp
        }))
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`Failed to save messages: ${response.status} - ${errorText}`)
    }

    console.log(`Saved ${messages.length} messages for thread ${threadId}`)
  } catch (error) {
    console.error('Error saving thread messages:', error)
    throw error
  }
}

/**
 * Fetch messages for a thread from MongoDB
 */
export async function fetchThreadMessages(threadId: string): Promise<Message[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/threads/${threadId}/messages`)

    if (!response.ok) {
      if (response.status === 404) {
        return []
      }
      const errorText = await response.text()
      throw new Error(`Failed to fetch messages: ${response.status} - ${errorText}`)
    }

    const data = await response.json()
    
    // Convert timestamp strings back to Date objects
    return (data.messages || []).map((msg: any) => ({
      id: msg.id,
      role: msg.role,
      content: msg.content,
      timestamp: msg.timestamp instanceof Date ? msg.timestamp : new Date(msg.timestamp)
    }))
  } catch (error) {
    console.error('Error fetching thread messages:', error)
    throw error
  }
}

/**
 * Save a single message to a thread
 */
export async function saveMessage(
  threadId: string,
  message: Message
): Promise<void> {
  // Use appendMessage for single messages
  return appendMessage(threadId, message)
}

/**
 * Get all threads
 */
export async function getAllThreads(): Promise<ThreadData[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/threads`)

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`Failed to fetch threads: ${response.status} - ${errorText}`)
    }

    const data = await response.json()
    return (data.threads || []).map((thread: any) => ({
      thread_id: thread.thread_id,
      messages: (thread.messages || []).map((msg: any) => ({
        id: msg.id,
        role: msg.role,
        content: msg.content,
        timestamp: msg.timestamp instanceof Date ? msg.timestamp : new Date(msg.timestamp)
      })),
      created_at: thread.created_at instanceof Date ? thread.created_at : new Date(thread.created_at),
      updated_at: thread.updated_at instanceof Date ? thread.updated_at : new Date(thread.updated_at)
    }))
  } catch (error) {
    console.error('Error fetching threads:', error)
    throw error
  }
}

/**
 * Delete a thread from MongoDB
 */
export async function deleteThread(threadId: string): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/threads/${threadId}`, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`Failed to delete thread: ${response.status} - ${errorText}`)
    }

    console.log(`Deleted thread: ${threadId}`)
  } catch (error) {
    console.error('Error deleting thread:', error)
    throw error
  }
}

