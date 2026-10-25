/** Minimal typings for the AudioWorkletGlobalScope (not part of lib.dom). */

export interface ProcessorBase {
  readonly port: MessagePort;
}

interface ProcessorCtor {
  new (options?: unknown): ProcessorBase;
}

interface WorkletScope {
  AudioWorkletProcessor: ProcessorCtor;
  registerProcessor(name: string, ctor: unknown): void;
  sampleRate: number;
  currentTime: number;
  currentFrame: number;
}

export const scope = globalThis as unknown as WorkletScope;
export const Processor: ProcessorCtor = scope.AudioWorkletProcessor;
