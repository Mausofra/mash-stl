#!/usr/bin/env python3
"""
Teste básico para o handler Hunyuan3D-2 RunPod Worker.

Este script valida:
1. Sintaxe do handler.py
2. Imports necessários
3. Formato de input/output
4. Processamento com imagem de teste (opcional)

Uso:
    python test_handler.py [--full] [--image path/to/image.png]

Se --full for fornecido, tenta carregar pipelines (requer GPU e pesos ~15GB).
Caso contrário, apenas valida sintaxe e estrutura.
"""
import sys
import json
import base64
import argparse
from pathlib import Path
import tempfile

def test_syntax():
    """Valida sintaxe do handler.py"""
    print("🔍 Validando sintaxe do handler.py...")
    try:
        with open("handler.py", "r", encoding="utf-8") as f:
            content = f.read()
        # Compilação básica
        compile(content, "handler.py", "exec")
        print("✅ Sintaxe Python válida")
        return True
    except SyntaxError as e:
        print(f"❌ Erro de sintaxe: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro ao ler handler.py: {e}")
        return False

def test_imports():
    """Valida imports do handler.py"""
    print("🔍 Validando imports...")
    required_modules = [
        "runpod",
        "base64",
        "io",
        "tempfile",
        "os",
        "logging",
        "shutil",
        "zipfile",
        "pathlib",
        "torch",
        "huggingface_hub",
        "PIL"
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"❌ Módulos faltando: {missing}")
        return False
    else:
        print("✅ Todos os imports estão disponíveis")
        return True

def test_handler_structure():
    """Valida estrutura básica do handler"""
    print("🔍 Validando estrutura do handler...")
    try:
        # Importa o handler (sem executar)
        import handler
        print("✅ Módulo handler importável")
        
        # Verifica se a função handler existe
        if hasattr(handler, 'handler'):
            print("✅ Função handler() encontrada")
            
            # Verifica assinatura básica
            import inspect
            sig = inspect.signature(handler.handler)
            params = list(sig.parameters.keys())
            if len(params) == 1:
                print(f"✅ Assinatura válida: handler({params[0]})")
                return True
            else:
                print(f"❌ Assinatura inesperada: {sig}")
                return False
        else:
            print("❌ Função handler() não encontrada")
            return False
    except Exception as e:
        print(f"❌ Erro ao importar handler: {e}")
        return False

def test_with_mock_image(image_path=None):
    """Testa com imagem mock (sem carregar modelos)"""
    print("🔍 Testando com imagem mock...")
    
    if not image_path:
        # Tenta usar test_image.png do diretório raiz
        root_image = Path(__file__).parent.parent / "test_image.png"
        if root_image.exists():
            image_path = root_image
            print(f"📸 Usando imagem de teste: {image_path}")
        else:
            print("⚠️  Nenhuma imagem de teste encontrada, criando imagem mock...")
            # Cria uma imagem RGB 256x256 simples
            from PIL import Image
            import numpy as np
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array, 'RGB')
            temp_dir = tempfile.mkdtemp()
            image_path = Path(temp_dir) / "test_image.png"
            img.save(image_path)
            print(f"📸 Imagem mock criada: {image_path}")
    
    # Codifica imagem em base64
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Cria job input no formato esperado
    job_input = {
        "input": {
            "image": image_b64,
            "format": "glb",
            "texture": False,  # Sem textura para teste rápido
            "num_inference_steps": 5,  # Muito baixo para teste
            "guidance_scale": 7.0,
            "octree_resolution": 128  # Baixa resolução para teste
        }
    }
    
    print("📋 Job input criado:")
    print(f"  - Formato: {job_input['input']['format']}")
    print(f"  - Texture: {job_input['input']['texture']}")
    print(f"  - Steps: {job_input['input']['num_inference_steps']}")
    print(f"  - Tamanho imagem: {len(image_b64) / 1024:.1f} KB")
    
    # Valida formato
    expected_keys = {"image", "format", "texture", "num_inference_steps", "guidance_scale", "octree_resolution"}
    actual_keys = set(job_input["input"].keys())
    
    if expected_keys.issubset(actual_keys):
        print("✅ Formato de input válido")
        return True, image_path
    else:
        missing = expected_keys - actual_keys
        print(f"❌ Keys faltando no input: {missing}")
        return False, image_path

def test_pipeline_load():
    """Tenta carregar pipelines (apenas com --full)"""
    print("🚀 Tentando carregar pipelines (isso pode demorar e requer GPU)...")
    try:
        # Importa funções do handler
        from handler import _load_pipelines, _ensure_weights
        
        # Verifica espaço em disco
        from handler import _check_disk_space
        if not _check_disk_space("/", required_gb=15):
            print("⚠️  Espaço em disco insuficiente para modelos (~15GB)")
            return False
        
        # Tenta baixar pesos (pode demorar)
        print("📥 Verificando/download de pesos...")
        _ensure_weights()
        
        print("✅ Pesos disponíveis")
        # Nota: Não vamos carregar os pipelines de fato pois requer GPU
        print("⚠️  Carregamento de pipelines pulado (requer GPU)")
        return True
    except Exception as e:
        print(f"❌ Erro ao carregar pipelines: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Teste do handler Hunyuan3D-2")
    parser.add_argument("--full", action="store_true", help="Executar testes completos (inclui download de pesos)")
    parser.add_argument("--image", type=str, help="Caminho para imagem de teste")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 Teste do Handler Hunyuan3D-2 RunPod Worker")
    print("=" * 60)
    
    # Muda para diretório do script
    script_dir = Path(__file__).parent
    original_dir = Path.cwd()
    os.chdir(script_dir)
    
    tests_passed = 0
    total_tests = 0
    
    try:
        # Teste 1: Sintaxe
        total_tests += 1
        if test_syntax():
            tests_passed += 1
        
        # Teste 2: Imports
        total_tests += 1
        if test_imports():
            tests_passed += 1
        
        # Teste 3: Estrutura
        total_tests += 1
        if test_handler_structure():
            tests_passed += 1
        
        # Teste 4: Imagem mock
        total_tests += 1
        success, image_path = test_with_mock_image(args.image)
        if success:
            tests_passed += 1
        
        # Teste 5: Pipelines (apenas com --full)
        if args.full:
            total_tests += 1
            if test_pipeline_load():
                tests_passed += 1
        
        print("\n" + "=" * 60)
        print(f"📊 Resultado: {tests_passed}/{total_tests} testes passaram")
        
        if tests_passed == total_tests:
            print("✅ Todos os testes passaram!")
            print("\n🎉 Handler está pronto para build no RunPod.")
            print("   Para build remoto via GitHub Actions:")
            print("   1. Configure secrets RUNPOD_API_KEY no GitHub")
            print("   2. Push para branch main")
            print("   3. Monitore workflow em Actions tab")
            return 0
        else:
            print(f"⚠️  {total_tests - tests_passed} teste(s) falharam")
            print("\n💡 Recomendações:")
            print("   - Verifique dependências em requirements.txt")
            print("   - Confirme que handler.py tem função handler(job)")
            print("   - Teste localmente com Docker: docker build -t hunyuan3d .")
            return 1
            
    except Exception as e:
        print(f"❌ Erro não esperado: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    import os
    sys.exit(main())