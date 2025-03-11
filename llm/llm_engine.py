import subprocess

def generate_response(prompt, max_tokens=2048):
    """Generates a response from Mistral 7B using Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral", prompt],
            capture_output=True,
            text=True,
            timeout=60  # Prevents infinite hangs
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"⚠️ Error in Ollama execution: {result.stderr}"
    except Exception as e:
        return f"⚠️ Exception while running Ollama: {str(e)}"

if __name__ == "__main__":
    # ✅ Quick Test
    test_prompt = "Explain the importance of dataset metadata."
    print("🧠 LLM Response:", generate_response(test_prompt))
