import express from 'express'
import cors from 'cors'
import { MongoClient } from 'mongodb'
import dotenv from 'dotenv'

dotenv.config()

const app = express()
const PORT = process.env.PORT || 3001

// Middleware
app.use(cors())
app.use(express.json())

// MongoDB connection
const MONGODB_URI = process.env.MONGODB_URI

if (!MONGODB_URI) {
  console.error('MONGODB_URI environment variable is not set')
  console.error('Please create a .env file in the backend directory with: MONGODB_URI=your_connection_string')
  throw new Error('MONGODB_URI environment variable is not set')
}

console.log('MongoDB URI loaded:', MONGODB_URI ? `${MONGODB_URI.substring(0, 20)}...` : 'NOT SET')

const DB_NAME = 'langchain_challenge'
const COLLECTION_NAME = 'threads'

let client = null
let db = null

async function connectToMongoDB() {
  try {
    if (!client || !db) {
      console.log('Connecting to MongoDB...')
      if (client) {
        try {
          await client.close()
        } catch (e) {
          // Ignore close errors
        }
      }
      client = new MongoClient(MONGODB_URI)
      await client.connect()
      db = client.db(DB_NAME)
      console.log(`Connected to MongoDB database: ${DB_NAME}`)
    }
    
    return db
  } catch (error) {
    console.error('MongoDB connection error:', error)
    // Reset connection on error
    client = null
    db = null
    throw error
  }
}

// Routes

// Create a new thread
app.post('/api/threads', async (req, res) => {
  try {
    const database = await connectToMongoDB()
    
    if (!database) {
      return res.status(500).json({ error: 'Database connection failed' })
    }
    
    const collection = database.collection(COLLECTION_NAME)
    const { thread_id } = req.body

    if (!thread_id) {
      return res.status(400).json({ error: 'thread_id is required' })
    }

    const now = new Date()
    await collection.insertOne({
      thread_id,
      messages: [],
      created_at: now,
      updated_at: now
    })

    res.json({ success: true, thread_id })
  } catch (error) {
    console.error('Error creating thread:', error)
    res.status(500).json({ error: error.message })
  }
})

// Get messages for a thread
app.get('/api/threads/:threadId/messages', async (req, res) => {
  try {
    const database = await connectToMongoDB()
    
    if (!database) {
      return res.status(500).json({ error: 'Database connection failed' })
    }
    
    const collection = database.collection(COLLECTION_NAME)
    const { threadId } = req.params

    const thread = await collection.findOne({ thread_id: threadId })

    if (!thread) {
      return res.json({ messages: [] })
    }

    res.json({
      messages: (thread.messages || []).map(msg => ({
        id: msg.id,
        role: msg.role,
        content: msg.content,
        timestamp: msg.timestamp
      }))
    })
  } catch (error) {
    console.error('Error fetching messages:', error)
    res.status(500).json({ error: error.message })
  }
})

// Save/append messages to a thread
app.post('/api/threads/:threadId/messages', async (req, res) => {
  try {
    const database = await connectToMongoDB()
    
    if (!database) {
      return res.status(500).json({ error: 'Database connection failed' })
    }
    
    const collection = database.collection(COLLECTION_NAME)
    const { threadId } = req.params
    const { messages } = req.body

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'messages array is required' })
    }

    const now = new Date()

    // Check if thread exists
    const existingThread = await collection.findOne({ thread_id: threadId })

    if (existingThread) {
      // Append new messages
      await collection.updateOne(
        { thread_id: threadId },
        {
          $push: {
            messages: { $each: messages }
          },
          $set: {
            updated_at: now
          }
        }
      )
    } else {
      // Create new thread with messages
      await collection.insertOne({
        thread_id: threadId,
        messages,
        created_at: now,
        updated_at: now
      })
    }

    res.json({ success: true, messageCount: messages.length })
  } catch (error) {
    console.error('Error saving messages:', error)
    res.status(500).json({ error: error.message })
  }
})

// Get all threads
app.get('/api/threads', async (req, res) => {
  try {
    const database = await connectToMongoDB()
    
    if (!database) {
      return res.status(500).json({ error: 'Database connection failed' })
    }
    
    const collection = database.collection(COLLECTION_NAME)

    const threads = await collection.find({}).sort({ updated_at: -1 }).toArray()

    res.json({
      threads: threads.map(thread => ({
        thread_id: thread.thread_id,
        messages: thread.messages || [],
        created_at: thread.created_at,
        updated_at: thread.updated_at
      }))
    })
  } catch (error) {
    console.error('Error fetching threads:', error)
    res.status(500).json({ error: error.message })
  }
})

// Delete a thread
app.delete('/api/threads/:threadId', async (req, res) => {
  try {
    const database = await connectToMongoDB()
    
    if (!database) {
      return res.status(500).json({ error: 'Database connection failed' })
    }
    
    const collection = database.collection(COLLECTION_NAME)
    const { threadId } = req.params

    const result = await collection.deleteOne({ thread_id: threadId })

    if (result.deletedCount === 0) {
      return res.status(404).json({ error: 'Thread not found' })
    }

    res.json({ success: true, thread_id: threadId })
  } catch (error) {
    console.error('Error deleting thread:', error)
    res.status(500).json({ error: error.message })
  }
})

// Health check
app.get('/health', async (req, res) => {
  try {
    const database = await connectToMongoDB()
    if (database) {
      res.json({ status: 'ok', mongodb: 'connected' })
    } else {
      res.status(500).json({ status: 'error', mongodb: 'disconnected' })
    }
  } catch (error) {
    res.status(500).json({ status: 'error', mongodb: 'error', error: error.message })
  }
})

// Test MongoDB connection on startup
async function testConnection() {
  try {
    await connectToMongoDB()
    console.log('MongoDB connection test: SUCCESS')
  } catch (error) {
    console.error('MongoDB connection test: FAILED')
    console.error('Error:', error.message)
  }
}

app.listen(PORT, async () => {
  console.log(`Backend server running on http://localhost:${PORT}`)
  await testConnection()
})

