## Windows Setup Instructions (PowerShell)

1. Install Miniconda:
   ```powershell
   iex ((New-Object System.Net.WebClient).DownloadString('https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.ps1'))
   ```

2. Create and activate environment:
   ```powershell
   conda env create -f environment.yml
   conda activate magentra
   ```

3. Additional Windows configuration:
   ```powershell
   conda install -c conda-forge vs2022_runtime
   pip install pywin32
   ```

4. Verify installation:
   ```powershell
   python -c "import langchain; print(f'LangChain {langchain.__version__} installed')"
   ``` 


   test command:  uvicorn backend.app:app --reload --app-dir backend 

  
  # ingest command

   curl -X POST "http://localhost:8000/ingest" ^
-H "Content-Type: application/json" ^
-d "{\"file_path\": \"D:\\MagentRA\\knowledge\\sample.txt\"}"

  # query command

curl -X POST "http://localhost:8000/query" ^
-H "Content-Type: application/json" ^
-d "{\"question\": \"Neo4j vector features\", \"k\": 3}"
 
 # Try different question types
curl -X POST "http://localhost:8000/query" ^
-H "Content-Type: application/json" ^
-d "{\"question\": \"What is the recommended chunk size?\", \"k\": 3}"
 


   # Try different k values
   curl -X POST "http://localhost:8000/query" ^
   -H "Content-Type: application/json" ^
   -d "{\"question\": \"What is the recommended chunk size?\", \"k\": 5}" 


