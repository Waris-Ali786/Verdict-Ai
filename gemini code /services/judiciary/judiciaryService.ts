import { ai, fileToBase64 } from "../common/geminiClient";
import { SignalData, CaseRecord } from "../../types";
import { calculatePriorityScore } from "../../lib/priorityEngine";

/**
 * JUDICIARY MODULE
 * This module handles case classification and priority scoring for the judiciary.
 */

export async function extractSignals(text: string): Promise<SignalData> {
  const model = "gemini-3-flash-preview";
  const snippet = text.slice(0, 5000);

  const prompt = `
    You are a STRICT legal classifier for Pakistani court case files.
    Your job is to EXTRACT and INFER missing information from the provided text.
    
    Rules:
    - Always infer case_type from text (use keywords like murder, terrorism, kidnapping, rape, robbery, fraud, drug, corruption, civil, property).
    - Always generate a realistic case_title if not present.
    - If court not mentioned, assume "Sessions Court".
    - If accused name not found, write "Unnamed Accused".
    - Do NOT leave important fields empty or unknown.
    
    Return ONLY valid JSON matching this structure:
    {
      "case_title": "...",
      "case_type": "murder | terrorism | kidnapping | rape | robbery | fraud | drug | corruption | civil | property",
      "section": "...",
      "court": "...",
      "accused_name": "...",
      "accused_in_custody": boolean,
      "involves_minor": boolean,
      "involves_woman": boolean,
      "involves_elder": boolean,
      "adjournment_count": number,
      "days_waiting": number,
      "summary": "...",
      "urgency_keywords": ["...", "..."]
    }
    
    Case text:
    """${snippet}"""
  `;

  try {
    const response = await ai.models.generateContent({
      model,
      contents: prompt,
      config: {
        responseMimeType: "application/json"
      }
    });

    return JSON.parse(response.text || "{}");
  } catch (error) {
    console.error("Signal Extraction Error:", error);
    return {
      case_title: "Unknown Case",
      case_type: "other",
      summary: "Error extracting signals."
    };
  }
}

export async function processCasePDF(file: File): Promise<CaseRecord> {
  const base64Data = await fileToBase64(file);
  const model = "gemini-3-flash-preview";

  const response = await ai.models.generateContent({
    model,
    contents: [
      {
        inlineData: {
          mimeType: "application/pdf",
          data: base64Data
        }
      },
      { text: "Extract all text from this PDF document." }
    ]
  });

  const text = response.text || "";
  const signals = await extractSignals(text);
  const priority = calculatePriorityScore(signals);

  return {
    id: Math.random().toString(36).substring(7),
    filename: file.name,
    title: signals.case_title || file.name.replace(".pdf", ""),
    signals,
    ...priority
  };
}
