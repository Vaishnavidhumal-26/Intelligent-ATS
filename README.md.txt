# Intelligent ATS — AI-Powered Resume Screening and Candidate Ranking System

## 1. Project Overview

Intelligent ATS is an AI-powered resume screening system that analyzes multiple resumes against a given Job Description (JD). It extracts resume text, identifies relevant skills, calculates semantic similarity, and ranks candidates based on a combined score.

The application supports PDF and DOCX resumes and provides a downloadable CSV report of candidate rankings.

## 2. Problem Statement

Manual resume screening can be time-consuming when recruiters receive many applications. Resumes may also use different words to describe similar skills and responsibilities.

This project aims to automate the initial screening process by combining skill matching with NLP-based semantic similarity.

## 3. Objectives

- Automate bulk resume screening.
- Extract text from PDF and DOCX resumes.
- Identify skills mentioned in resumes and job descriptions.
- Compare resumes with a job description using semantic similarity.
- Rank candidates using a combined scoring formula.
- Display matched and missing skills.
- Generate a downloadable candidate-ranking report.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Tabular data processing and CSV reports |
| PyMuPDF | Extracting text from PDF files |
| python-docx | Extracting text from DOCX files |
| Sentence Transformers | Generating semantic text embeddings |
| all-MiniLM-L6-v2 | Pre-trained NLP embedding model |
| Scikit-learn | Calculating cosine similarity |
| Streamlit | Building the web application |

## 5. Model Selection

### Selected Model: all-MiniLM-L6-v2

The project uses the pre-trained Sentence Transformer model `sentence-transformers/all-MiniLM-L6-v2`.

### Why was this model selected?

- It generates compact numerical representations (embeddings) of text.
- It can compare the semantic meaning of a job description and a resume.
- It is suitable for sentence and paragraph similarity tasks.
- It is relatively lightweight compared with many larger transformer models.
- It can run locally after the model files have been downloaded.

The model is used to calculate semantic similarity. It has not been fine-tuned on a custom resume dataset for this project.

### Model Storage

The model is stored locally in the project directory:

`models/all-MiniLM-L6-v2/`

The application loads the model from this local directory instead of requiring a fresh model download at every startup.

## 6. Project Workflow

1. Recruiter enters a job description.
2. Recruiter uploads multiple PDF or DOCX resumes.
3. The application extracts text from each resume.
4. Text is normalized for comparison.
5. Skills are identified from the predefined skill list.
6. Skill matching features are calculated.
7. Resume and job-description embeddings are generated.
8. Cosine similarity is calculated.
9. Skill and semantic scores are combined.
10. Candidates are sorted by final score.
11. Results are displayed and exported as a CSV report.

## 7. Feature Engineering

Feature engineering is used to convert resume and job-description text into useful comparison features.

### Features extracted

- **Resume text:** Extracted content from the uploaded file.
- **Skill list:** Skills detected using a predefined vocabulary.
- **Matched skills:** Skills found in both the resume and job description.
- **Missing skills:** Skills detected in the job description but not found in the resume.
- **Skill Match Score:** Percentage of identified job-description skills matched by the resume.
- **Semantic Similarity Score:** Similarity between the resume and job-description embeddings.
- **Resume Word Count:** Number of words in the extracted resume text.
- **Resume Text Length:** Number of characters in the normalized resume text.

Word count and text length are currently descriptive features; they are not included in the final ranking formula.

### Text preprocessing

- Convert text to lowercase for skill matching.
- Normalize repeated whitespace.
- Use word boundaries when detecting skills to reduce unintended partial matches.
- Handle unreadable or empty extracted text without ranking it as a normal candidate.

### Skill matching formula

Skill Match Score is calculated as:

Skill Match Score = (Number of Matched Job Skills / Number of Identified Job Skills) × 100

If no supported skills are detected in the job description, the skill score is set to zero. The result should be interpreted cautiously because the predefined vocabulary may not cover every relevant skill.

## 8. Semantic Similarity

The pre-trained model converts the job description and each resume into embeddings. Cosine similarity is used to estimate how closely the texts are related in meaning.

The raw cosine similarity is limited to the range 0 to 1 and then multiplied by 100 to obtain the semantic similarity score.

Semantic similarity helps compare related wording, but it does not establish that a candidate is qualified or has verified a particular skill.

## 9. Candidate Ranking Formula

The current application uses the following weighted formula:

Final Score = (0.50 × Skill Match Score) + (0.50 × Semantic Similarity Score)

Both component scores are represented on a 0–100 scale.

The skill and semantic components receive equal weight in the current implementation. The weights are configurable design choices and have not been optimized using a labeled recruitment dataset.

Candidates are sorted in descending order of their final score.

## 10. Data Quality and Outlier Handling

This project primarily processes unstructured text rather than a conventional numerical training dataset. Therefore, numerical outlier treatment methods such as IQR-based capping are not directly applicable to the text itself.

The current implementation includes the following basic data-quality checks:

- Skip resumes from which no readable text can be extracted.
- Handle extraction errors and report affected files.
- Report resume word count and text length for inspection.
- Restrict the semantic similarity score to the 0–100 range.
- Allow unreadable or scanned PDFs to be identified for further processing.

**Limitations and planned improvements:**

- Detect duplicate resumes using normalized text or document hashes.
- Identify unusually short or long extracted documents for manual review.
- Add OCR support for scanned PDF resumes.
- Investigate duplicated content caused by PDF extraction or repeated DOCX table content.
- Validate scoring behavior using a labeled evaluation dataset.

The current implementation does not yet perform a comprehensive duplicate-resume detection or statistical outlier-removal procedure. These are planned improvements, not completed features.

## 11. Application Features

- Bulk resume upload.
- PDF and DOCX text extraction.
- Job-description input.
- Skill matching and missing-skill identification.
- Semantic similarity calculation.
- Candidate ranking table.
- Candidate-level score and text details.
- CSV report download.
- Local pre-trained model loading.

## 12. Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version.
- pip.
- The project source code.
- The downloaded model files in `models/all-MiniLM-L6-v2/`.

### Install dependencies

Open a terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to use the application.

## 13. Project Structure

```text
Intelligent_ATS/
│
├── app.py
├── Intelligent_ATS.ipynb
├── README.md
├── requirements.txt
│
├── models/
│   └── all-MiniLM-L6-v2/
│
├── Data/
├── Report/
│   └── ATS_Candidate_Ranking.csv
│
└── Sample Resume/
```

The report file is generated when results are downloaded or exported. The model directory contains the local pre-trained model files.

## 14. Testing and Evaluation

The application should be tested with:

- PDF and DOCX resumes.
- Multiple resumes uploaded together.
- Empty or unreadable documents.
- Job descriptions containing known and unknown skills.
- Resumes with strong, partial, and weak skill matches.
- Repeated uploads of the same resume.
- Job descriptions with no detectable skills.
- Comparison of ranking results with manual review.

Useful evaluation measures for a future labeled dataset include Precision@K, Recall@K, ranking quality, and agreement with human-reviewed candidate relevance.

No formal benchmark accuracy or recruitment-performance claim is made without a suitable labeled evaluation dataset.

## 15. Limitations

- Skill extraction relies on a predefined skill vocabulary.
- Equivalent skills and abbreviations may not always be recognized.
- PDF extraction may fail for scanned documents.
- Semantic similarity does not verify skills, qualifications, or experience.
- The model has not been fine-tuned on recruitment-specific data.
- The current score weights have not been validated against hiring outcomes.
- Ranking can be affected by resume formatting and missing information.

The application is intended to assist initial screening, not to make final hiring decisions. Recruiters should review resumes and assess candidates fairly.

## 16. Future Enhancements

- OCR support for scanned resumes.
- Duplicate-resume detection.
- Improved skill synonym mapping.
- Experience and education extraction.
- Additional feature engineering and validated ranking weights.
- Evaluation using manually labeled resume–job-description pairs.
- Explainable candidate-ranking summaries.
- More extensive testing and error reporting.

## 17. Conclusion

Intelligent ATS demonstrates how text extraction, skill matching, semantic embeddings, cosine similarity, and a Streamlit interface can be combined to support bulk resume screening.

The project provides a foundation for more advanced NLP-based recruitment tools. Its scores should be validated and interpreted alongside human review before real-world hiring use.
