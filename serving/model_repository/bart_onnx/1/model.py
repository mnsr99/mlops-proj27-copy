import triton_python_backend_utils as pb_utils
import numpy as np
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

class TritonPythonModel:
    def initialize(self, args):
        model_path = "/workspace/onnx_model" 
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Using CUDAExecutionProvider to leverage the P100 GPU
        self.model = ORTModelForSeq2SeqLM.from_pretrained(
            model_path, 
            provider="CUDAExecutionProvider" 
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            input_texts = [val.decode("utf-8") for val in in_tensor.as_numpy().flatten().tolist()]

            # Tokenize
            inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True)

            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            # Generate summary/output
            outputs = self.model.generate(**inputs, max_length=150, min_length=30)
            
            # Decode
            decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            out_tensor = pb_utils.Tensor(
                "OUTPUT_TEXT", np.array(decoded_outputs, dtype=object)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
            
        return responses
