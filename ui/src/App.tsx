import { useState } from 'react'
import './App.css'
import ChatTab from './components/ChatTab'
import MonitoringTab from './components/MonitoringTab'

type Tab = 'chat' | 'monitoring'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('chat')

  return (
    <div className="app-container">
      {/* Vertical Navbar */}
      <nav className="vertical-navbar">
        <button
          className={`nav-button ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          <span className="nav-text">Chatting</span>
        </button>
        <button
          className={`nav-button ${activeTab === 'monitoring' ? 'active' : ''}`}
          onClick={() => setActiveTab('monitoring')}
        >
          <span className="nav-text">Monitoring</span>
        </button>
      </nav>

      {/* Main Content Area */}
      <main className="main-content">
        {activeTab === 'chat' && <ChatTab />}
        {activeTab === 'monitoring' && <MonitoringTab />}
      </main>
    </div>
  )
}

export default App
