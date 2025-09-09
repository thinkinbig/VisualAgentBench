import React from 'react';
import { TrajectorySummary } from '../types';
import { Play, Calendar, Target, GitBranch, CheckCircle } from 'lucide-react';

interface TrajectoryListProps {
  trajectories: TrajectorySummary[];
  selectedRunId: string | null;
  onSelectTrajectory: (runId: string) => void;
  loading: boolean;
}

const TrajectoryList: React.FC<TrajectoryListProps> = ({
  trajectories,
  selectedRunId,
  onSelectTrajectory,
  loading
}) => {
  if (loading) {
    return (
      <div className="trajectory-list">
        <div className="trajectory-list-header">
          <h2>Trajectories</h2>
        </div>
        <div className="loading">Loading trajectories...</div>
      </div>
    );
  }

  if (trajectories.length === 0) {
    return (
      <div className="trajectory-list">
        <div className="trajectory-list-header">
          <h2>Trajectories</h2>
        </div>
        <div className="empty-state">
          <Play size={48} className="empty-icon" />
          <p>No trajectories found</p>
          <small>Run some tasks to generate trajectory data</small>
        </div>
      </div>
    );
  }

  return (
    <div className="trajectory-list">
      <div className="trajectory-list-header">
        <h2>Trajectories</h2>
        <span className="count">{trajectories.length} total</span>
      </div>
      
      <div className="trajectory-items">
        {trajectories.map((trajectory) => (
          <div
            key={trajectory.filename}
            className={`trajectory-item ${selectedRunId === trajectory.filename ? 'selected' : ''}`}
            onClick={() => onSelectTrajectory(trajectory.filename)}
          >
            <div className="trajectory-header">
              <div className="trajectory-title">
                <Target size={16} />
                <span className="run-id">{trajectory.run_id}</span>
              </div>
              <div className="trajectory-date">
                <Calendar size={14} />
                <span>{new Date(trajectory.created_at).toLocaleString()}</span>
              </div>
            </div>
            
            <div className="trajectory-intent">
              {trajectory.intent}
            </div>
            
            <div className="trajectory-stats">
              <div className="stat">
                <GitBranch size={14} />
                <span>{trajectory.total_nodes} nodes</span>
              </div>
              <div className="stat">
                <div className="stat-dot state" />
                <span>{trajectory.state_nodes} states</span>
              </div>
              <div className="stat">
                <div className="stat-dot candidate" />
                <span>{trajectory.candidate_nodes} candidates</span>
              </div>
              <div className="stat">
                <CheckCircle size={14} />
                <span>{trajectory.selected_nodes} selected</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TrajectoryList;
