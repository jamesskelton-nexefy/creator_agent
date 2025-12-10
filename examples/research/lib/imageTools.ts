/**
 * Image Generation Tools
 * 
 * LangChain tools for AI image generation using nano-banana-pro.
 * These tools can be used by agents to generate and manage images.
 */

import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { 
  generateImage as generateImageFromLib, 
  IMAGE_PRESETS,
  type ImagePreset,
  type AspectRatio,
  type OutputFormat,
  type GenerateImageResult,
} from "./imageGenerator";

// ============================================================================
// SCHEMA DEFINITIONS
// ============================================================================

/**
 * Zod schema for image generation parameters.
 */
const generateImageSchema = z.object({
  prompt: z
    .string()
    .describe("A detailed text description of the image to generate. Be specific about style, subject, colors, and composition."),
  preset: z
    .enum(["banner", "hero", "content", "thumbnail", "square", "portrait", "custom"])
    .optional()
    .describe(
      'eLearning preset determining aspect ratio: "banner" (21:9 for course headers), "hero" (16:9 for slides), "content" (16:9 general), "thumbnail" (3:2 for cards), "square" (1:1 for icons), "portrait" (3:4 for people), "custom" (specify aspectRatio)'
    ),
  aspectRatio: z
    .enum(["1:1", "16:9", "4:3", "3:2", "21:9", "9:16", "3:4", "2:3", "4:5", "5:4"])
    .optional()
    .describe('Custom aspect ratio. Only used when preset is "custom". Default: "16:9"'),
  outputFormat: z
    .enum(["png", "jpg"])
    .optional()
    .describe('Image format. Default: "png" for best quality'),
  title: z
    .string()
    .optional()
    .describe("Optional title for the generated image (used when storing)"),
  description: z
    .string()
    .optional()
    .describe("Optional description for metadata (helps with searchability in media library)"),
});

/**
 * Zod schema for listing presets.
 */
const listPresetsSchema = z.object({});

// ============================================================================
// TOOL DEFINITIONS
// ============================================================================

/**
 * Generate an AI image using nano-banana-pro.
 * 
 * This tool generates images optimized for eLearning content with
 * preset aspect ratios for common use cases like banners, hero images,
 * thumbnails, etc.
 */
export const generateImageTool = tool(
  async ({ prompt, preset, aspectRatio, outputFormat, title, description }): Promise<string> => {
    console.log(`[generateImage] Starting generation...`);
    console.log(`[generateImage] Preset: ${preset || "content"}, Format: ${outputFormat || "png"}`);

    try {
      const result = await generateImageFromLib({
        prompt,
        preset: (preset as ImagePreset) || "content",
        aspectRatio: aspectRatio as AspectRatio | undefined,
        outputFormat: (outputFormat as OutputFormat) || "png",
      });

      // Build metadata for storage
      const metadata = {
        ai_generated: true,
        prompt: result.prompt,
        model: "google/nano-banana-pro",
        aspect_ratio: result.aspectRatio,
        output_format: result.outputFormat,
        preset: result.preset,
        generated_at: result.generatedAt,
        description: description || undefined,
      };

      // Return result as JSON string for agent processing
      return JSON.stringify({
        success: true,
        message: `Image generated successfully`,
        image: {
          url: result.url,
          bufferSize: result.buffer.length,
          aspectRatio: result.aspectRatio,
          outputFormat: result.outputFormat,
          preset: result.preset,
        },
        metadata,
        // Include buffer as base64 for potential storage
        bufferBase64: result.buffer.toString("base64"),
        suggestedTitle: title || `AI Generated - ${result.preset || "custom"}`,
      });
    } catch (error) {
      console.error("[generateImage] Error:", error);
      return JSON.stringify({
        success: false,
        error: `Image generation failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
    }
  },
  {
    name: "generateImage",
    description: `Generate an AI image using Google's nano-banana-pro model. 
    
Use for creating eLearning visuals like:
- Course banners and headers (preset: "banner")
- Hero images and slides (preset: "hero")  
- Content illustrations (preset: "content")
- Card thumbnails (preset: "thumbnail")
- Icons and avatars (preset: "square")
- Portrait images (preset: "portrait")

The generated image URL and buffer are returned. Use the buffer to upload to storage.`,
    schema: generateImageSchema,
  }
);

/**
 * List available image generation presets.
 * 
 * Returns information about each preset including aspect ratio,
 * description, and common use cases.
 */
export const listImagePresetsTool = tool(
  async (): Promise<string> => {
    const presets = Object.entries(IMAGE_PRESETS).map(([name, config]) => ({
      name,
      aspectRatio: config.aspectRatio,
      description: config.description,
      useCases: config.useCases,
    }));

    return JSON.stringify({
      success: true,
      presets,
      supportedFormats: ["png", "jpg"],
      supportedAspectRatios: [
        "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
      ],
    });
  },
  {
    name: "listImagePresets",
    description: "List available image generation presets with their aspect ratios and use cases. Use this to understand which preset is best for your needs.",
    schema: listPresetsSchema,
  }
);

// ============================================================================
// EXPORTS
// ============================================================================

/**
 * All image generation tools for use with agents.
 */
export const imageTools = [
  generateImageTool,
  listImagePresetsTool,
];

export default imageTools;



