import axios from 'axios';
import { TrajectorySummary, TrajectoryDetail, GraphvizResponse } from './types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

export const trajectoryApi = {
  // Get list of all trajectories
  getTrajectories: async (): Promise<TrajectorySummary[]> => {
    const response = await api.get('/trajectories');
    return response.data;
  },

  // Get detailed trajectory by filename
  getTrajectory: async (filename: string): Promise<TrajectoryDetail> => {
    const response = await api.get(`/trajectories/${filename}`);
    return response.data;
  },

  // Get Graphviz DOT source
  getGraphviz: async (filename: string): Promise<GraphvizResponse> => {
    const response = await api.get(`/trajectories/${filename}/graphviz`);
    return response.data;
  },

  // Get screenshot URL
  getScreenshotUrl: (filename: string): string => {
    return `${API_BASE_URL}/screenshots/${filename}`;
  },

  // Health check
  healthCheck: async (): Promise<{ status: string; timestamp: string }> => {
    const response = await api.get('/health');
    return response.data;
  },
};
