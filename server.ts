import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import cors from 'cors';
import {
  CopilotRuntime,
  AnthropicAdapter,
  copilotRuntimeNodeHttpEndpoint,
  LangGraphAgent
} from '@copilotkit/runtime';
import Anthropic from '@anthropic-ai/sdk';
import { createDocumentService, isValidDocumentType, getFileTypeFromMime } from './src/documents/index';

const app = express();

// Enable CORS for frontend
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:5173', 'http://127.0.0.1:3000', 'http://127.0.0.1:5173'],
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));

// Parse JSON bodies - increased limit for large CopilotKit payloads
// (conversations with many messages, tool results with document content)
// Note: CopilotKit's endpoint uses GraphQL Yoga which has its own body parser,
// but this covers other routes and provides a fallback
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));
// Raw body parser for routes that need unparsed bodies (like GraphQL)
app.use(express.raw({ limit: '50mb', type: 'application/json' }));

// Configure multer for file uploads (memory storage for processing)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 100 * 1024 * 1024, // 100MB limit
  },
  fileFilter: (_req, file, cb) => {
    if (isValidDocumentType(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only PDF and DOCX files are supported.'));
    }
  },
});


// Verify API key exists
if (!process.env.ANTHROPIC_API_KEY) {
  console.error('[FATAL] ANTHROPIC_API_KEY not found in environment variables');
  process.exit(1);
}

console.log('[√] ANTHROPIC_API_KEY found');

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const serviceAdapter = new AnthropicAdapter({
  anthropic,
  model: 'claude-sonnet-4-20250514',
});

console.log('[√] Anthropic service adapter initialized');
console.log('    Model: claude-sonnet-4-20250514');

// ============================================================================
// DOCUMENT API ENDPOINTS
// ============================================================================

// Initialize document service lazily
let documentService: ReturnType<typeof createDocumentService> | null = null;

function getDocumentService() {
  if (!documentService) {
    documentService = createDocumentService();
  }
  return documentService;
}

/**
 * POST /api/documents/upload
 * Upload and process a document (PDF or DOCX)
 */
app.post('/api/documents/upload', (req, res, next) => {
  upload.single('file')(req, res, (err) => {
    if (err) {
      if (err instanceof multer.MulterError) {
        if (err.code === 'LIMIT_FILE_SIZE') {
          return res.status(413).json({
            success: false,
            error: 'File too large. Maximum file size is 100MB.',
          });
        }
        return res.status(400).json({
          success: false,
          error: `Upload error: ${err.message}`,
        });
      }
      return res.status(400).json({
        success: false,
        error: err.message || 'Upload failed',
      });
    }
    next();
  });
}, async (req, res) => {
  console.log('\n[Documents API] Upload request received');
  
  try {
    const file = req.file;
    if (!file) {
      res.status(400).json({ success: false, error: 'No file provided' });
      return;
    }

    const { title, category, orgId, projectId } = req.body;

    // Validate required fields
    if (!orgId) {
      res.status(400).json({ success: false, error: 'orgId is required' });
      return;
    }
    if (!category || !['course_content', 'framework_content'].includes(category)) {
      res.status(400).json({ success: false, error: 'category must be "course_content" or "framework_content"' });
      return;
    }

    const fileType = getFileTypeFromMime(file.mimetype);
    if (!fileType) {
      res.status(400).json({ success: false, error: 'Invalid file type' });
      return;
    }

    console.log(`  File: ${file.originalname} (${file.size} bytes)`);
    console.log(`  Type: ${fileType}`);
    console.log(`  Category: ${category}`);
    console.log(`  OrgId: ${orgId}`);

    // Process the document
    const result = await getDocumentService().uploadDocument({
      buffer: file.buffer,
      filename: file.originalname,
      mimeType: file.mimetype,
      orgId,
      projectId: projectId || undefined,
      category: category as 'course_content' | 'framework_content',
      title: title || undefined,
    });

    console.log(`  [√] Document processed: ${result.documentId}`);

    res.json({
      success: true,
      documentId: result.documentId,
      title: result.title,
      totalLines: result.totalLines,
      totalChunks: result.totalChunks,
      status: result.status,
    });
  } catch (error) {
    console.error('[Documents API] Upload error:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * GET /api/documents
 * List all documents
 */
app.get('/api/documents', async (req, res) => {
  try {
    const { category, orgId, projectId, limit, offset } = req.query;

    const docs = await getDocumentService().listDocuments({
      category: category as 'course_content' | 'framework_content' | undefined,
      orgId: orgId as string | undefined,
      projectId: projectId as string | undefined,
      limit: limit ? parseInt(limit as string) : undefined,
      offset: offset ? parseInt(offset as string) : undefined,
    });

    res.json({
      success: true,
      count: docs.length,
      documents: docs,
    });
  } catch (error) {
    console.error('[Documents API] List error:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * GET /api/documents/:id
 * Get a single document by ID
 */
app.get('/api/documents/:id', async (req, res) => {
  try {
    const doc = await getDocumentService().getDocument(req.params.id);

    if (!doc) {
      res.status(404).json({ success: false, error: 'Document not found' });
      return;
    }

    res.json({
      success: true,
      document: doc,
    });
  } catch (error) {
    console.error('[Documents API] Get error:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * DELETE /api/documents/:id
 * Delete a document
 */
app.delete('/api/documents/:id', async (req, res) => {
  try {
    await getDocumentService().deleteDocument(req.params.id);

    res.json({
      success: true,
      message: 'Document deleted',
    });
  } catch (error) {
    console.error('[Documents API] Delete error:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * POST /api/documents/search
 * Search documents (semantic or text)
 */
app.post('/api/documents/search', async (req, res) => {
  try {
    const { query, searchText, category, orgId, projectId, limit, threshold } = req.body;

    if (!query && !searchText) {
      res.status(400).json({ success: false, error: 'Either query (semantic) or searchText (exact) is required' });
      return;
    }

    let results;
    if (query) {
      // Semantic search
      results = await getDocumentService().searchDocuments({
        query,
        category,
        orgId,
        projectId,
        limit,
        threshold,
      });
    } else {
      // Text search
      results = await getDocumentService().searchDocumentsByText({
        searchText,
        category,
        orgId,
        projectId,
        limit,
      });
    }

    res.json({
      success: true,
      count: results.length,
      searchType: query ? 'semantic' : 'text',
      results,
    });
  } catch (error) {
    console.error('[Documents API] Search error:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

console.log('[√] Document API endpoints configured');
console.log('    POST /api/documents/upload');
console.log('    GET  /api/documents');
console.log('    GET  /api/documents/:id');
console.log('    DELETE /api/documents/:id');
console.log('    POST /api/documents/search');

// ============================================================================
// COPILOTKIT RUNTIME
// ============================================================================

// ============================================================================
// LANGGRAPH DEPLOYMENT CONFIGURATION
// ============================================================================
// Set USE_LOCAL_LANGGRAPH=true to use local langgraph server (localhost:8000)
// Otherwise uses LangSmith Cloud deployment
const USE_LOCAL_LANGGRAPH = process.env.USE_LOCAL_LANGGRAPH === 'true';
const LANGGRAPH_CLOUD_URL = 'https://creatoragentcloud-9617e637415c5949bf1f5028e5c0361c.us.langgraph.app';
const LANGGRAPH_LOCAL_URL = 'http://localhost:8000';
const LANGGRAPH_URL = USE_LOCAL_LANGGRAPH ? LANGGRAPH_LOCAL_URL : LANGGRAPH_CLOUD_URL;

// LangSmith API key required for cloud deployment
const LANGSMITH_API_KEY = process.env.LANGSMITH_API_KEY;
if (!USE_LOCAL_LANGGRAPH && !LANGSMITH_API_KEY) {
  console.warn('[WARNING] LANGSMITH_API_KEY not set - cloud deployment may fail authentication');
}

// Select which orchestrator to use: 'orchestrator' (LangGraph) or 'orchestrator_langchain' (createAgent)
const ORCHESTRATOR_AGENT = process.env.ORCHESTRATOR_AGENT || 'orchestrator';

console.log('[√] Configuring CopilotKit agents...');
console.log(`    [AGENT LOCK MODE] Locked to: ${ORCHESTRATOR_AGENT}`);
console.log(`    Mode: ${USE_LOCAL_LANGGRAPH ? 'LOCAL' : 'CLOUD'}`);
console.log(`    Agent: ${ORCHESTRATOR_AGENT} @ ${LANGGRAPH_URL}`);

app.use('/copilotkit', (req, res, next) => {
  // ============================================================================
  // MULTI-AGENT ORCHESTRATOR MODE
  // ============================================================================
  // CopilotKit is configured to work with the orchestrator agent which
  // coordinates specialized sub-agents: Strategist, Researcher, Architect,
  // Writer, and Visual Designer.
  // ============================================================================
  const runtime = new CopilotRuntime({
    // =========================================================================
    // AGENT LOCK MODE - Force ALL requests through the orchestrator agent
    // =========================================================================
    // The orchestrator coordinates between specialized sub-agents:
    // - Strategist: Requirements gathering
    // - Researcher: Deep knowledge gathering  
    // - Architect: Course structure design
    // - Writer: Content creation
    // - Visual Designer: Aesthetics and tone
    // =========================================================================
    agentLock: ORCHESTRATOR_AGENT,
    agents: {
      'orchestrator': new LangGraphAgent({
        deploymentUrl: LANGGRAPH_URL,
        langsmithApiKey: LANGSMITH_API_KEY,
        graphId: 'orchestrator',
        description: `Multi-agent orchestrator for creating impactful online training:
- Coordinates specialized sub-agents for different tasks
- Strategist: Discovers purpose, objectives, scope
- Researcher: Gathers deep knowledge on topics
- Architect: Designs course structure
- Writer: Creates Level 6 content nodes
- Visual Designer: Defines aesthetics and tone`,
        // Note: LangGraph Platform API accepts both snake_case and camelCase
        // Including both for compatibility
        config: {
          recursion_limit: 150,
          recursionLimit: 150,
        },
      }),
      'orchestrator_langchain': new LangGraphAgent({
        deploymentUrl: LANGGRAPH_URL,
        langsmithApiKey: LANGSMITH_API_KEY,
        graphId: 'orchestrator_langchain',
        description: `LangChain createAgent orchestrator - a simpler agent implementation:
- Uses LangChain v1 createAgent pattern
- Standard ReAct agent loop with automatic tool handling
- Access to all frontend tools for project/node management
- May provide more stable tool handling than custom LangGraph`,
        config: {
          recursion_limit: 150,
          recursionLimit: 150,
        },
      }),
    },
    middleware: {
      onBeforeRequest: (options) => {
        const startTime = Date.now();
        console.log('\n[CopilotKit] ===== Request started =====');
        console.log('  Thread ID:', options.threadId);
        console.log('  Run ID:', options.runId);
        console.log('  Messages:', options.inputMessages.length, 'message(s)');
        if (options.inputMessages.length > 0) {
          const lastMsg = options.inputMessages[options.inputMessages.length - 1];
          console.log('  Last message preview:', JSON.stringify(lastMsg).substring(0, 150) + '...');
        }
        console.log('  Properties:', JSON.stringify(options.properties, null, 2));
        
        // Log if CopilotKit Cloud API key is present
        if (options.properties?.copilotCloudPublicApiKey) {
          console.log('  [Observability] CopilotKit Cloud API key detected - traces will be sent');
        } else {
          console.log('  [Observability] No CopilotKit Cloud API key - traces disabled');
        }
        
        // Set a timeout warning
        const timeoutWarning = setTimeout(() => {
          console.warn(`\n[WARNING] Request ${options.threadId} taking longer than 10 seconds...`);
          console.warn('  This might indicate a hang in router or agent selection');
        }, 10000);
        
        // Store for cleanup
        (options as any)._timeoutWarning = timeoutWarning;
        (options as any)._startTime = startTime;
      },
      onAfterRequest: (options) => {
        const duration = Date.now() - ((options as any)._startTime || Date.now());
        clearTimeout((options as any)._timeoutWarning);
        
        console.log('[CopilotKit] ----- Request completed -----');
        console.log('  Thread ID:', options.threadId);
        console.log('  Duration:', duration, 'ms');
        console.log('  Input messages:', options.inputMessages.length);
        console.log('  Output messages:', options.outputMessages.length);
        if (options.outputMessages.length > 0) {
          const lastOut = options.outputMessages[options.outputMessages.length - 1];
          console.log('  Last output preview:', JSON.stringify(lastOut).substring(0, 150) + '...');
        }
        console.log('======================================\n');
      },
    },
  });

  const handler = copilotRuntimeNodeHttpEndpoint({
    endpoint: '/copilotkit',
    runtime,
    serviceAdapter,
  });

  return handler(req, res, next);
});

app.listen(4000, () => {
  console.log('\n' + '='.repeat(60));
  console.log('[√] CopilotKit Runtime Server Started');
  console.log('='.repeat(60));
  console.log('  URL: http://localhost:4000/copilotkit');
  console.log(`  Active Agent: ${ORCHESTRATOR_AGENT}`);
  if (ORCHESTRATOR_AGENT === 'orchestrator') {
    console.log('  Sub-agents: strategist, researcher, architect, writer, visual_designer');
  } else {
    console.log('  Mode: LangChain createAgent (simpler ReAct loop)');
  }
  console.log(`  LangGraph: ${LANGGRAPH_URL}`);
  console.log(`  Mode: ${USE_LOCAL_LANGGRAPH ? 'LOCAL' : 'LANGSMITH CLOUD'}`);
  console.log('  Model: claude-sonnet-4-20250514');
  console.log('  [TIP] Set ORCHESTRATOR_AGENT=orchestrator_langchain to use createAgent version');
  console.log('='.repeat(60) + '\n');
});