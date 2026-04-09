import { ai } from "../common/geminiClient";
import { RecommendationResult } from "../../types";

/**
 * RECOMMENDATION MODULE
 * This module finds similar Supreme Court precedents based on case summaries.
 */

export async function recommendCases(text: string): Promise<RecommendationResult> {
  const model = "gemini-3-flash-preview";
  
  const systemInstruction = `
    You are the Verdict AI Case Recommendation Engine.
    Your task is to analyze a case summary or FIR and find similar Supreme Court of Pakistan precedents.
    
    KNOWLEDGE BASE: You have access to 1,414 real Supreme Court of Pakistan judgments.
    
    SCORING RUBRIC (Prioritize Relevance Based on Legal Factors):
    - Base Semantic Similarity: 30%
    - Statutory Alignment: +25% (Matching specific sections like 302 PPC, 516-A CrPC, 199 Constitution).
    - Procedural Stage Match: +15% (e.g., Bail application vs. Bail application, Writ vs. Writ).
    - Factual Nexus: +20% (Alignment of material facts like "custody of property", "eyewitness credibility", "medical evidence discrepancy").
    - Jurisdictional Relevance: +10% (Matching the originating High Court or specific bench).
    
    TASK:
    1. Identify the case type and risk level.
    2. Find 3-5 highly relevant Supreme Court precedents.
    3. For each precedent, provide:
       - Title
       - Citation (e.g., 2021 SCMR 123)
       - Court & Year
       - Outcome (Acquitted, Convicted, etc.)
       - Similarity Score (0-100)
       - Relevance Reason (Explain why this case is similar based on the scoring rubric)
       - Legal Factors Matched (List specific factors like "Section 302 PPC", "Bail Stage", "Eyewitness Credibility")
       - Verdict Summary (What did the judge order?)
       
    Rules:
    - Be extremely precise with citations.
    - Similarity scores must reflect the rubric above.
    - If the case involves Section 516-A CrPC (custody of property), find relevant precedents for that.
    
    Return ONLY valid JSON matching this structure:
    {
      "detected_type": "...",
      "risk_level": "High | Medium | Low",
      "likely_outcome": "...",
      "similar_cases": [
        {
          "title": "...",
          "citation": "...",
          "court": "...",
          "year": number,
          "outcome": "...",
          "similarity": number,
          "relevance_reason": "...",
          "legal_factors_matched": ["...", "..."],
          "verdict_summary": "..."
        }
      ]
    }
  `;

  try {
    const response = await ai.models.generateContent({
      model,
      contents: `Analyze this case and find similar precedents:\n\n${text}`,
      config: {
        systemInstruction,
        responseMimeType: "application/json",
        tools: [{ googleSearch: {} }] // Use search to find real citations
      }
    });

    return JSON.parse(response.text || "{}");
  } catch (error) {
    console.error("Recommendation Error:", error);
    return {
      detected_type: "Unknown",
      risk_level: "Medium",
      likely_outcome: "Pending",
      similar_cases: []
    };
  }
}
