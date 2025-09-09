export interface TrajectorySummary {
  filename: string;
  run_id: string;
  intent: string;
  total_nodes: number;
  state_nodes: number;
  candidate_nodes: number;
  selected_nodes: number;
  created_at: string;
  file_path: string;
}

export interface NodeInfo {
  node_id: string;
  parent_id: string | null;
  node_type: 'root' | 'state' | 'candidate';
  step: number | null;
  url: string | null;
  thought: string | null;
  action: string | null;
  meaning: string | null;
  status: string | null;
  screenshot_path: string | null;
  candidates: string[];
}

export interface TrajectoryDetail {
  run_id: string;
  intent: string;
  nodes: NodeInfo[];
  created_at: string;
}

export interface GraphvizResponse {
  dot_source: string;
}
