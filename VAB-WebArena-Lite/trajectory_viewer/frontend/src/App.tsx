import React, { useState, useEffect } from 'react';
import { TrajectorySummary, TrajectoryDetail } from './types';
import { trajectoryApi } from './api';
import TrajectoryList from './components/TrajectoryList';
import TrajectoryVisualization from './components/TrajectoryVisualization';
import { RefreshCw, AlertCircle } from 'lucide-react';
import './App.css';

const App: React.FC = () => {
  const [trajectories, setTrajectories] = useState<TrajectorySummary[]>([]);
  const [selectedTrajectory, setSelectedTrajectory] = useState<TrajectoryDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadTrajectories = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await trajectoryApi.getTrajectories();
      setTrajectories(data);
    } catch (err) {
      setError('Failed to load trajectories. Make sure the backend server is running.');
      console.error('Error loading trajectories:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectTrajectory = async (runId: string) => {
    try {
      setError(null);
      const trajectory = await trajectoryApi.getTrajectory(runId);
      setSelectedTrajectory(trajectory);
    } catch (err) {
      setError('Failed to load trajectory details.');
      console.error('Error loading trajectory:', err);
    }
  };

  const handleCloseVisualization = () => {
    setSelectedTrajectory(null);
  };

  useEffect(() => {
    loadTrajectories();
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>Trajectory Tree Viewer</h1>
          <p>Interactive visualization of agent execution trajectories</p>
        </div>
        <button 
          onClick={loadTrajectories} 
          className="refresh-button"
          disabled={loading}
          title="Refresh trajectories"
        >
          <RefreshCw size={20} className={loading ? 'spinning' : ''} />
        </button>
      </header>

      <main className="app-main">
        {error && (
          <div className="error-banner">
            <AlertCircle size={20} />
            <span>{error}</span>
            <button onClick={loadTrajectories} className="retry-button">
              Retry
            </button>
          </div>
        )}

        <div className="app-content">
          <TrajectoryList
            trajectories={trajectories}
            selectedRunId={selectedTrajectory?.run_id || null}
            onSelectTrajectory={handleSelectTrajectory}
            loading={loading}
          />

          {selectedTrajectory && (
            <TrajectoryVisualization
              trajectory={selectedTrajectory}
              onClose={handleCloseVisualization}
            />
          )}
        </div>
      </main>
    </div>
  );
};

export default App;
