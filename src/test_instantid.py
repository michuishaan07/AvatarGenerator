from insightface.app import FaceAnalysis

# Auto-downloads AntelopeV2 to ~/.insightface/models
app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)
