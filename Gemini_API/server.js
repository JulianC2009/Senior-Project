import express from 'express';
import { GoogleGenerativeAI } from "@google/generative-ai";
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.join(__dirname, '.env') });

const app = express();
app.use(express.json());

console.log("Checking API Key:", process.env.GEMINI_API_KEY ? "Loaded" : "Not Found");

app.use(express.static(path.join(__dirname, 'public')));

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "YOUR_BACKUP_KEY_HERE"); 
const model = genAI.getGenerativeModel({ 
    model: "gemini-2.0-flash",
    systemInstruction: "You are a helpful Health Assistant. Be concise and professional." 
});

app.post('/api/chat', async (req, res) => {
    try {
        const { message } = req.body;
        console.log("User said:", message); // Logs the incoming message

        const result = await model.generateContent(message);
        const response = await result.response;
        const text = response.text();
        
        console.log("Gemini replied successfully!");
        res.json({ reply: text });
    } catch (error) {
        // Prints the error to the terminal
        console.error("--- GEMINI ERROR ---");
        console.error(error.message); 
        console.error("--------------------");
        
        res.status(500).json({ error: "Gemini API Error: " + error.message });
    }
});

app.listen(3000, () => {
    console.log('Server is running at http://localhost:3000');
});