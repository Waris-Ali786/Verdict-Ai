export type UserRole = 'user' | 'lawyer' | 'deskaid';

export interface User {
  name: string;
  email: string;
  role: UserRole;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  urduContent?: string;
  englishContent?: string;
  timestamp: number;
  audioUrl?: string;
}

export interface Session {
  id: string;
  name: string;
  messages: Message[];
  createdAt: number;
}

export type LawyerFeature = 'drafting' | 'case-law' | 'understanding-laws' | 'chat' | 'summarizer' | 'priority-engine' | 'case-recommendation' | 'citation-verification';

export interface CitationVerificationResult {
  citation: string;
  status: 'Valid' | 'Overruled' | 'Distinguished' | 'Caution' | 'Unknown';
  title: string;
  court: string;
  year: number;
  link?: string;
  summary: string;
  subsequent_history: {
    citation: string;
    treatment: string;
    reason: string;
  }[];
  analysis: string;
}

export interface SignalData {
  case_title?: string;
  case_type?: string;
  section?: string;
  court?: string;
  accused_name?: string;
  accused_in_custody?: boolean;
  involves_minor?: boolean;
  involves_woman?: boolean;
  involves_elder?: boolean;
  adjournment_count?: number;
  days_waiting?: number;
  summary?: string;
  urgency_keywords?: string[];
}

export interface ScoreBreakdown {
  signal: string;
  detail: string;
  points: number;
  max: number;
}

export interface PriorityResult {
  score: number;
  tag: 'Critical' | 'Medium' | 'Routine';
  breakdown: ScoreBreakdown[];
}

export interface CaseRecord extends PriorityResult {
  id: string;
  filename: string;
  title: string;
  signals: SignalData;
}

export interface RecommendedCase {
  title: string;
  citation: string;
  court: string;
  year: number;
  outcome: string;
  similarity: number;
  relevance_reason: string;
  verdict_summary: string;
  legal_factors_matched: string[];
}

export interface RecommendationResult {
  detected_type: string;
  risk_level: 'High' | 'Medium' | 'Low';
  likely_outcome: string;
  similar_cases: RecommendedCase[];
}
