import { ai, fileToBase64 } from "../common/geminiClient";
import { CitationVerificationResult } from "../../types";

/**
 * LEGAL RESEARCH MODULE
 * This module handles citation verification and document summarization.
 */

// In-memory store for document context (base64 data)
const documentContexts = new Map<string, { data: string, filename: string, mimeType: string }>();

export async function summarizeDocument(file: File) {
  try {
    const base64Data = await fileToBase64(file);
    const fileId = Math.random().toString(36).substring(7);
    
    documentContexts.set(fileId, {
      data: base64Data,
      filename: file.name,
      mimeType: file.type || "application/pdf"
    });

    const prompt = `
      You are a professional legal assistant in Pakistan.
      Provide a concise and clear summary of this legal document or verdict.
      Focus on:
      1. Key Facts
      2. The Decision/Judgment
      3. Specific Legal Statutes or Precedents mentioned.
      
      Ensure the tone is formal and professional.
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: [
        {
          inlineData: {
            mimeType: file.type || "application/pdf",
            data: base64Data,
          },
        },
        { text: prompt },
      ],
    });

    return { summary: response.text || "Could not generate summary.", fileId };
  } catch (error: any) {
    console.error("Summarization Error:", error);
    return { 
      summary: `I'm sorry, I encountered an error while summarizing the document: ${error.message || 'Unknown error'}`,
      fileId: null
    };
  }
}

export async function ask_question(fileId: string, question: string) {
  try {
    const context = documentContexts.get(fileId);
    if (!context) {
      throw new Error("Document context not found. Please re-upload.");
    }

    const prompt = `
      You are a helpful legal assistant. 
      Based ONLY on the attached legal document (${context.filename}), answer the following question:
      
      Question: ${question}
      
      If the answer is not in the document, say "I don't know based on the provided text."
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: [
        {
          inlineData: {
            mimeType: context.mimeType,
            data: context.data,
          },
        },
        { text: prompt },
      ],
    });

    return { answer: response.text || "I'm sorry, I couldn't generate an answer." };
  } catch (error) {
    console.error("QA Error:", error);
    return { answer: "I'm sorry, I couldn't process your question about this document." };
  }
}

export async function verifyCitation(citation: string): Promise<CitationVerificationResult> {
  const model = "gemini-3-flash-preview";
  
  const systemInstruction = `
    You are the Verdict AI Citation Verifier.
    Your task is to verify a Supreme Court or High Court of Pakistan legal citation.
    
    TASK:
    1. Identify the case title, court, and year from the citation.
    2. Provide a direct URL to the case law on the official Supreme Court or High Court website (e.g., supremecourt.gov.pk, lahorehighcourt.gov.pk, etc.).
    3. Determine the CURRENT STATUS of the case (Valid, Overruled, Distinguished, Caution).
    4. Provide a brief summary of the original judgment.
    5. Find SUBSEQUENT HISTORY: List cases that have cited, overruled, or distinguished this case.
    6. Provide a final legal analysis of whether it is safe to rely on this citation in court today.
    
    Rules:
    - Use Google Search to find the most recent status and the official link.
    - Be extremely precise with subsequent citations.
    - If the case is overruled, clearly state by which judgment (Citation and Title).
    
    Return ONLY valid JSON matching this structure:
    {
      "citation": "...",
      "status": "Valid | Overruled | Distinguished | Caution | Unknown",
      "title": "...",
      "court": "...",
      "year": number,
      "link": "https://...",
      "summary": "...",
      "subsequent_history": [
        {
          "citation": "...",
          "treatment": "Overruled | Distinguished | Followed | Cited",
          "reason": "..."
        }
      ],
      "analysis": "..."
    }
  `;

  try {
    const response = await ai.models.generateContent({
      model,
      contents: `Verify this legal citation: ${citation}`,
      config: {
        systemInstruction,
        responseMimeType: "application/json",
        tools: [{ googleSearch: {} }]
      }
    });

    return JSON.parse(response.text || "{}");
  } catch (error) {
    console.error("Citation Verification Error:", error);
    return {
      citation,
      status: "Unknown",
      title: "Unknown",
      court: "Unknown",
      year: 0,
      summary: "Error verifying citation.",
      subsequent_history: [],
      analysis: "Could not verify citation due to an error."
    };
  }
}
