// ir_fuzz.cpp — libFuzzer entry point for the runtime dialect IR parser
// TODO: implement on Linux with MLIR + Clang

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // TODO: implement on Linux with MLIR
    // std::string ir_text(reinterpret_cast<const char*>(data), size);
    // mlir::MLIRContext ctx;
    // ctx.loadDialect<runtime::RuntimeDialect>();
    // auto module = mlir::parseSourceString<mlir::ModuleOp>(ir_text, &ctx);
    // if (module) {
    //   mlir::PassManager pm(&ctx);
    //   pm.addPass(createShapeInferencePass());
    //   pm.addPass(createFusionPass());
    //   (void)pm.run(*module);  // ignore failures — we're fuzzing for crashes
    // }
    return 0;
}
