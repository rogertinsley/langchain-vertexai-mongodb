# MongoDB, VertexAI, LangChain

```bash
docker compose up
npm install
```
Create a `.env` file in the root directory
  
```
MONGODB_ATLAS_URI=your_mongodb_atlas_uri_here
```

```bash
npm run seed
npm run dev
```

Start a new conversation:
```
curl -X POST -H "Content-Type: application/json" -d '{"message": "Your message here"}' http://localhost:3000/chat
```

Continue an existing conversation:
```
curl -X POST -H "Content-Type: application/json" -d '{"message": "Your follow-up message"}' http://localhost:3000/chat/{threadId}
```

## How it works

1. The seed script in `seed-database.ts` generates synthetic employee data and populates the MongoDB database.
2. The LangGraph agent is defined in `agent.ts`, including the conversation graph structure and tools.
3. MongoDB operations are integrated directly into the agent for storing and retrieving conversation data.
4. The Express server in `index.ts` provides API endpoints for starting and continuing conversations.
5. User inputs are processed through the LangGraph agent, generating appropriate responses and updating the conversation state.
6. Conversation data is persisted in MongoDB, allowing for continuity across sessions.
