import { ai } from "../common/geminiClient";

/**
 * TRANSCRIPTION MODULE
 * This module handles audio transcription for court statements.
 */

export async function transcribeAudio(audioBase64: string) {
  const model = "gemini-3-flash-preview"; 
  
  const systemInstruction = `
    ROLE: You are Verdict AI - Expert Court Transcriptionist for the Pakistani legal system.
    TASK: Analyze the provided audio. 
    1. Detect if the language is Urdu or English.
    2. If Urdu is spoken:
       - Transcribe the original Urdu script (Urdu Mode).
       - Provide a formal, professional English translation (English Mode).
    3. If English is spoken:
       - Transcribe the English text.
       - Provide the same text for both Urdu and English modes (since it's already English).
    4. FORMAT: You MUST return a JSON object with two keys: "urdu" and "english".
    5. Ensure the Urdu script is accurate and the English translation is formal/legal grade.
    6. If the audio is unclear, state "Unclear audio" in both fields.
  `;

  try {
    const response = await ai.models.generateContent({
      model,
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: "audio/webm",
              data: audioBase64
            }
          },
          { text: "Transcribe this legal statement." }
        ]
      },
      config: {
        systemInstruction,
        responseMimeType: "application/json"
      },
    });
    
    return JSON.parse(response.text || "{}");
  } catch (error) {
    console.error("Transcription Error:", error);
    return { urdu: "Error in transcription", english: "Error in transcription" };
  }
}
