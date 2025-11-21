import { MongoClient, Db, Collection } from 'mongodb'

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

// MongoDB connection
// Note: In Vite, environment variables must be prefixed with VITE_ to be accessible
const MONGODB_URI = import.meta.env.VITE_MONGODB_URI

if (!MONGODB_URI) {
  throw new Error('VITE_MONGODB_URI environment variable is not set')
}

const DB_NAME = 'langchain_chat'
const COLLECTION_NAME = 'threads'

let client: MongoClient | null = null
let db: Db | null = null

/**
 * Get MongoDB client connection
 */
async function getClient(): Promise<MongoClient> {
  if (!client) {
    client = new MongoClient(MONGODB_URI)
    await client.connect()
    console.log('Connected to MongoDB')
  }
  return client
}

/**
 * Get database instance
 */
async function getDb(): Promise<Db> {
  if (!db) {
    const mongoClient = await getClient()
    db = mongoClient.db(DB_NAME)
  }
  return db
}

/**
 * Get threads collection
 */
async function getCollection(): Promise<Collection> {
  const database = await getDb()
  return database.collection(COLLECTION_NAME)
}

/**
 * Create a new thread document in MongoDB
 * Called when user starts a new conversation
 */
export async function createThread(threadId: string): Promise<void> {
  try {
    const collection = await getCollection()
    const now = new Date()

    await collection.insertOne({
      thread_id: threadId,
      messages: [],
      created_at: now,
      updated_at: now
    })

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
    const collection = await getCollection()
    const now = new Date()

    const result = await collection.updateOne(
      { thread_id: threadId },
      {
        $push: {
          messages: {
            id: message.id,
            role: message.role,
            content: message.content,
            timestamp: message.timestamp
          } as any
        },
        $set: {
          updated_at: now
        }
      }
    )

    if (result.matchedCount === 0) {
      // Thread doesn't exist, create it first then add message
      await createThread(threadId)
      await appendMessage(threadId, message)
      return
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
    const collection = await getCollection()
    const now = new Date()

    await collection.updateOne(
      { thread_id: threadId },
      {
        $set: {
          thread_id: threadId,
          messages: messages.map(msg => ({
            id: msg.id,
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp
          })),
          updated_at: now
        },
        $setOnInsert: {
          created_at: now
        }
      },
      { upsert: true }
    )

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
    const collection = await getCollection()
    const thread = await collection.findOne({ thread_id: threadId })

    if (!thread) {
      return []
    }

    // Convert timestamp strings back to Date objects
    return (thread.messages || []).map((msg: any) => ({
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
  try {
    const collection = await getCollection()
    const now = new Date()

    await collection.updateOne(
      { thread_id: threadId },
      {
        $push: {
          messages: {
            id: message.id,
            role: message.role,
            content: message.content,
            timestamp: message.timestamp
          } as any
        },
        $set: {
          updated_at: now
        },
        $setOnInsert: {
          created_at: now
        }
      },
      { upsert: true }
    )

    console.log(`Saved message to thread ${threadId}`)
  } catch (error) {
    console.error('Error saving message:', error)
    throw error
  }
}

/**
 * Get all threads (optional - if you want to list threads)
 */
export async function getAllThreads(): Promise<ThreadData[]> {
  try {
    const collection = await getCollection()
    const threads = await collection.find({}).sort({ updated_at: -1 }).toArray()

    return threads.map((thread: any) => ({
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
 * Close MongoDB connection
 */
export async function closeConnection(): Promise<void> {
  if (client) {
    await client.close()
    client = null
    db = null
    console.log('MongoDB connection closed')
  }
}

