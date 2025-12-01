import { useState, useEffect } from 'react'
import './App.css'
import ChatTab from './components/ChatTab'
import MonitoringTab from './components/MonitoringTab'
import { SettingsModal, type AgentConfig } from './components/SettingsModal'

type Tab = 'chat' | 'monitoring'

const defaultConfig: AgentConfig = {
  search_api: 'tavily',
  max_search_results: 5,
  max_research_depth: 3,
  max_subtopics: 7,
  max_clarification_rounds: 0,
  min_credibility_score: 0.3,
}

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('chat')
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [agentConfig, setAgentConfig] = useState<AgentConfig>(defaultConfig)

  // Load configuration from localStorage on mount
  useEffect(() => {
    const savedConfig = localStorage.getItem('agentConfig')
    if (savedConfig) {
      try {
        const parsed = JSON.parse(savedConfig)
        // Sanitize config: ensure search_api is 'tavily' (not 'serper')
        if (parsed.search_api && parsed.search_api !== 'tavily') {
          parsed.search_api = 'tavily'
          // Update localStorage with corrected value
          localStorage.setItem('agentConfig', JSON.stringify({ ...defaultConfig, ...parsed }))
        }
        setAgentConfig({ ...defaultConfig, ...parsed })
      } catch (e) {
        console.error('Failed to parse saved config:', e)
      }
    }
  }, [])

  const handleConfigSave = (config: AgentConfig) => {
    setAgentConfig(config)
    localStorage.setItem('agentConfig', JSON.stringify(config))
  }

  return (
    <div className="app-container">
      {/* Vertical Navbar */}
      <nav className="vertical-navbar">
        <div className="nav-buttons-top">
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
        </div>
        <div className="nav-buttons-bottom">
          <button
            className="nav-button settings-nav-button"
            onClick={() => setIsSettingsOpen(true)}
          >
            <span className="nav-text">Settings</span>
          </button>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="main-content">
        {activeTab === 'chat' && <ChatTab agentConfig={agentConfig} />}
        {activeTab === 'monitoring' && <MonitoringTab />}
      </main>

      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onSave={handleConfigSave}
        currentConfig={agentConfig}
      />
    </div>
  )
}

export default App
