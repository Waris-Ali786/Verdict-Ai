import { SignalData, PriorityResult, ScoreBreakdown } from '../types';

export function calculatePriorityScore(signals: SignalData): PriorityResult {
  let score = 0;
  const breakdown: ScoreBreakdown[] = [];

  // 1. Case type weight (max 35 pts)
  const typeWeights: Record<string, number> = {
    "terrorism": 35,
    "murder": 30,
    "rape": 28,
    "kidnapping": 28,
    "drug": 22,
    "robbery": 20,
    "corruption": 18,
    "fraud": 15,
    "civil": 10,
    "property": 8,
    "other": 5,
  };

  const caseType = (signals.case_type || "other").toLowerCase();
  const typePts = typeWeights[caseType] || 5;
  score += typePts;
  breakdown.push({
    signal: "Case type",
    detail: caseType.charAt(0).toUpperCase() + caseType.slice(1),
    points: typePts,
    max: 35,
  });

  // 2. Time waiting (max 25 pts)
  const days = parseInt(String(signals.days_waiting || 0));
  let waitPts = 2;
  if (days > 365) waitPts = 25;
  else if (days > 180) waitPts = 18;
  else if (days > 90) waitPts = 10;
  else if (days > 30) waitPts = 5;
  
  score += waitPts;
  breakdown.push({
    signal: "Time waiting",
    detail: `${days} days pending`,
    points: waitPts,
    max: 25,
  });

  // 3. Accused in custody (max 20 pts)
  if (signals.accused_in_custody) {
    score += 20;
    breakdown.push({
      signal: "Accused in custody",
      detail: "Liberty at stake — expedite hearing",
      points: 20,
      max: 20,
    });
  }

  // 4. Vulnerable persons (max 15 pts)
  if (signals.involves_minor) {
    score += 15;
    breakdown.push({
      signal: "Minor involved",
      detail: "Child victim or accused",
      points: 15,
      max: 15,
    });
  } else if (signals.involves_woman) {
    score += 10;
    breakdown.push({
      signal: "Woman involved",
      detail: "Female victim or accused",
      points: 10,
      max: 15,
    });
  } else if (signals.involves_elder) {
    score += 8;
    breakdown.push({
      signal: "Elderly involved",
      detail: "Senior citizen victim or accused",
      points: 8,
      max: 15,
    });
  }

  // 5. Adjournments (max 5 pts)
  const adj = parseInt(String(signals.adjournment_count || 0));
  let adjPts = 0;
  if (adj > 7) adjPts = 5;
  else if (adj > 3) adjPts = 3;
  else if (adj > 1) adjPts = 1;

  if (adjPts > 0) {
    score += adjPts;
    breakdown.push({
      signal: "Adjournments",
      detail: `${adj} delays recorded`,
      points: adjPts,
      max: 5,
    });
  }

  const finalScore = Math.min(score, 100);
  let tag: "Critical" | "Medium" | "Routine" = "Routine";
  if (finalScore >= 75) tag = "Critical";
  else if (finalScore >= 40) tag = "Medium";

  return {
    score: finalScore,
    tag,
    breakdown
  };
}
