import express from "express";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();
console.log("KEY PRESENT:", !!process.env.OPENAI_API_KEY);
console.log("KEY PREFIX:", (process.env.OPENAI_API_KEY || "").slice(0, 7));


const app = express();
app.use(express.json());
app.use(express.static("public"));

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

app.post("/api/chat", async (req, res) => {
  try {
    const { message } = req.body;

    if (!message) {
      return res.status(400).json({ error: "Missing 'message' in request body." });
    }

    const response = await client.responses.create({
      model: "gpt-4.1-mini",
      input: [
        { role: "system", content: "You are a helpful chatbot." },
        { role: "user", content: message }
      ]
    });

    res.json({ reply: response.output_text || "" });
  } catch (err) {
  console.error("OpenAI error:", err?.status, err?.message);
  console.error(err?.response?.data || err); // extra detail of error if present
  res.status(500).json({ error: err?.message || "OpenAI request failed." });
}
});

app.listen(3000, () => {
  console.log("Server running at http://localhost:3000");
});
