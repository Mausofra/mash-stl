#!/usr/bin/env python3
"""
CI/CD Sanity Test para o Hunyuan3D-2.1 RunPod Worker.
Este script é feito para rodar dentro do Dockerfile durante o build.
Ele utiliza "Mocks" para validar a lógica do handler sem precisar de GPU ou internet.
"""
import sys
import base64
import tempfile
import inspect
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mock apenas do que não está instalado no CI (torch e huggingface_hub)
# botocore/boto3 estão instalados e não devem ser mockados
for _mod in ["torch", "torch.cuda", "huggingface_hub"]:
    sys.modules[_mod] = MagicMock()

def run_tests():
    print("🧪 Iniciando bateria de testes estruturais do Handler...\n")

    # 1. Teste de Sintaxe e Importação
    print("1️⃣ Testando Sintaxe e Imports...")
    try:
        import handler
        print("✅ Importação bem-sucedida (Sem erros de sintaxe ou dependências faltando).")
    except Exception as e:
        print(f"❌ Erro fatal ao importar o handler: {e}")
        return False

    # 2. Teste de Assinatura da Função
    print("\n2️⃣ Testando Assinatura do Endpoint...")
    sig = inspect.signature(handler.handler)
    if "job" in sig.parameters:
        print("✅ Função principal 'handler(job)' validada.")
    else:
        print("❌ A função handler não aceita o parâmetro obrigatório 'job'.")
        return False

    # 3. Execução Seca (Dry-Run) com Mocks
    print("\n3️⃣ Testando Lógica de Processamento (Dry-Run)...")
    
    # Cria uma imagem fake minúscula (10x10 pixels vermelhos) para o teste
    try:
        from PIL import Image
        import io
        img = Image.new('RGBA', (10, 10), color=(255, 0, 0, 255))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        fake_b64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"❌ Falha ao gerar imagem de teste: {e}")
        return False

    # Payload simulado incluindo parâmetros da v2.1
    job_input = {
        "input": {
            "images": [fake_b64, fake_b64],   # multi-view: 2 ângulos
            "prompt": "a small red cube",
            "format": "obj",
            "texture": True,
            "num_inference_steps": 10,
            "guidance_scale": 7.0,
            "octree_resolution": 256,
        }
    }

    # Interceptando as funções pesadas/externas
    with patch('handler._load_shape'), \
         patch('handler._unload_shape'), \
         patch('handler._load_paint'), \
         patch('handler._check_disk_space', return_value=True), \
         patch('handler._upload_to_r2_and_get_url', return_value="https://fake-r2-url.com/model.zip"), \
         patch('handler.SHAPE_PIPELINE') as mock_shape, \
         patch('handler.PAINT_PIPELINE') as mock_paint, \
         patch('handler._export_mesh') as mock_export:

        # Simulando o comportamento da IA
        mock_shape.return_value = [MagicMock()]
        mock_paint.return_value = MagicMock()

        # Simulando a exportação do arquivo 3D
        fake_out_path = Path(tempfile.gettempdir()) / "fake_output.zip"
        fake_out_path.write_text("conteudo zip dummy")
        mock_export.return_value = fake_out_path

        try:
            result = handler.handler(job_input)

            if "error" in result:
                print(f"❌ Handler falhou na validação interna: {result['error']}")
                return False

            # Validações específicas do handler atualizado
            if result.get("format") == "zip" and "mesh_url" in result:
                print("✅ Decodificação de imagem executada.")
                print("✅ Pipeline de IA acionado (mockado).")
                print("✅ Exportação para OBJ → ZIP validada.")
                print("✅ Upload para R2 simulado com sucesso.")
            else:
                print(f"❌ Resposta inesperada: {result}")
                return False

        except Exception as e:
            print(f"❌ Exceção durante execução do handler: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if fake_out_path.exists():
                fake_out_path.unlink()

    print("\n🎉 TODOS OS TESTES PASSARAM! O código é seguro para inicialização.")
    return True

if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)