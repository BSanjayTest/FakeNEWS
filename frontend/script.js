document.addEventListener('DOMContentLoaded', () => {
    // Keep references to static/button elements here
    const checkButton = document.getElementById('checkButton');
    const newsTextArea = document.getElementById('newsText');
    const resultArea = document.getElementById('resultArea');
    const errorArea = document.getElementById('errorArea');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorMessageP = document.getElementById('errorMessage');

    // Remove references to dynamic result elements from here
    // const analyzedTextSpan = ...
    // const predictionSpan = ...
    // const probabilitySpan = ...
    // const scrapingStatusSpan = ...
    // const scrapedLinksList = ...
    // const sentimentLabelSpan = ...
    // const sentimentScoreSpan = ...
    // const linguisticFlagsList = ...
    // const similarTrueList = ...
    // const similarFakeList = ...
    // const llmAnalysisTextP = ...

    checkButton.addEventListener('click', async () => {
        const text = newsTextArea.value.trim();

        // Get references to most dynamic result elements *inside* the handler
        const analyzedTextSpan = document.getElementById('analyzedText');
        const predictionSpan = document.getElementById('prediction');
        const probabilitySpan = document.getElementById('probability');
        const scrapingStatusSpan = document.getElementById('scrapingStatus');
        const scrapedLinksList = document.getElementById('scrapedLinksList');
        const sentimentLabelSpan = document.getElementById('sentimentLabel');
        const sentimentScoreSpan = document.getElementById('sentimentScore');
        const linguisticFlagsList = document.getElementById('linguisticFlagsList');
        const similarTrueList = document.getElementById('similarTrueList');
        const similarFakeList = document.getElementById('similarFakeList');

        // Get references for scraped analysis section
        const scrapedKeywordsP = document.getElementById('scrapedKeywords');
        const scrapedAnalysisSourcesList = document.getElementById('scrapedAnalysisSourcesList');

        // Check if elements are null (excluding llmAnalysisTextP for now)
        if (!analyzedTextSpan || !predictionSpan || !probabilitySpan || !scrapingStatusSpan ||
            !scrapedLinksList || !sentimentLabelSpan || !sentimentScoreSpan || !linguisticFlagsList ||
            !similarTrueList || !similarFakeList || !errorMessageP || !resultArea ||
            !scrapedKeywordsP || !scrapedAnalysisSourcesList ) { // Check new elements
            console.error("CRITICAL: One or more required display elements could not be found!");
            if(errorMessageP) errorMessageP.textContent = "Internal page error: Could not find result display elements.";
            if(errorArea) errorArea.style.display = 'block';
            return; // Stop processing
        }

        // Clear previous results/errors
        resultArea.style.display = 'none';
        errorArea.style.display = 'none';
        loadingIndicator.style.display = 'none';
        errorMessageP.textContent = '';
        scrapedLinksList.innerHTML = '';
        linguisticFlagsList.innerHTML = '';
        similarTrueList.innerHTML = '';
        similarFakeList.innerHTML = '';
        sentimentLabelSpan.textContent = 'N/A';
        sentimentScoreSpan.textContent = 'N/A';
        scrapedKeywordsP.textContent = 'Analyzing...';
        scrapedAnalysisSourcesList.innerHTML = '';

        if (!text) {
            errorMessageP.textContent = 'Please paste some news text into the text area.';
            errorArea.style.display = 'block';
            return;
        }

        checkButton.disabled = true;
        checkButton.textContent = 'Analyzing...';
        loadingIndicator.style.display = 'block';

        try {
            // Make API call to the backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();

            // Populate results
            analyzedTextSpan.textContent = result.analyzed_text.substring(0, 150);
            predictionSpan.textContent = result.classification || 'N/A';
            probabilitySpan.textContent = result.classification_confidence
                ? (result.classification_confidence * 100).toFixed(2)
                : 'N/A';
            sentimentLabelSpan.textContent = result.sentiment.label || 'Error';
            sentimentScoreSpan.textContent = result.sentiment.score !== undefined ? result.sentiment.score.toFixed(3) : 'N/A';
            populateSimpleList(linguisticFlagsList, result.basic_linguistic_flags || ["Analysis Error."]);
            populateSimpleList(similarTrueList, result.similarity_results.similar_true || ["No similar true articles found or error."]);
            populateSimpleList(similarFakeList, result.similarity_results.similar_fake || ["No similar fake articles found or error."]);
            scrapingStatusSpan.textContent = result.scraped_results.message || 'Status unknown';
            scrapedLinksList.innerHTML = ''; // Clear first
            if (result.scraped_results.status === 'success' && result.scraped_results.data && result.scraped_results.data.length > 0) {
                 result.scraped_results.data.forEach(item => {
                    const li = document.createElement('li');
                    const a = document.createElement('a');
                    a.href = item.link;
                    a.textContent = item.title;
                    a.target = '_blank';
                    a.rel = 'noopener noreferrer';
                    li.appendChild(a);
                    const span = document.createElement('span');
                    span.textContent = item.snippet;
                    li.appendChild(span);
                    scrapedLinksList.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.textContent = result.scraped_results.message || 'No related articles found or search failed.';
                if (result.scraped_results.status === 'error' || result.scraped_results.status === 'skipped') {
                     li.style.fontStyle = 'italic';
                     li.style.color = '#888';
                }
                scrapedLinksList.appendChild(li);
            }

            // 5b. Populate Scraped Analysis Results
            if (result.scraped_analysis && result.scraped_analysis.keywords) {
                scrapedKeywordsP.textContent = result.scraped_analysis.keywords.join(', ') || 'No specific keywords identified.';
            } else {
                scrapedKeywordsP.textContent = 'Keyword analysis not available or failed.';
            }
            scrapedAnalysisSourcesList.innerHTML = ''; // Clear list
            if (result.scraped_analysis && result.scraped_analysis.sources && result.scraped_analysis.sources.length > 0) {
                result.scraped_analysis.sources.forEach(item => {
                    const li = document.createElement('li');
                    const a = document.createElement('a');
                    a.href = item.link;
                    a.textContent = item.title;
                    a.target = '_blank';
                    a.rel = 'noopener noreferrer';
                    li.appendChild(a);
                    // Optionally add snippet span if desired
                    // const span = document.createElement('span');
                    // span.textContent = item.snippet;
                    // li.appendChild(span);
                    scrapedAnalysisSourcesList.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.textContent = 'No sources listed for analysis.';
                scrapedAnalysisSourcesList.appendChild(li);
            }

            // 6. LLM Analysis - Find element just before updating
            const llmAnalysisTextP = resultArea.querySelector('#llmAnalysisText');
            if (llmAnalysisTextP) {
                llmAnalysisTextP.textContent = result.llm_analysis || "LLM analysis was not returned or failed.";
            } else {
                console.error("Cannot display LLM analysis - element #llmAnalysisText not found within #resultArea.");
                 // Optionally display error to user if element missing
                // errorMessageP.textContent = "Internal page error: Cannot display LLM analysis.";
                // errorArea.style.display = 'block';
            }

            resultArea.style.display = 'block'; // Show results section

        } catch (error) {
            console.error('Error during analysis fetch or processing:', error);
            errorMessageP.textContent = `An error occurred: ${error.message}`;
            errorArea.style.display = 'block';
        } finally {
            checkButton.disabled = false;
            checkButton.textContent = 'Check News';
            loadingIndicator.style.display = 'none';
        }
    });
});

// Helper function to populate a simple list (ul element) with text items
function populateSimpleList(listElement, items) {
    listElement.innerHTML = ''; // Clear existing items
    if (items && items.length > 0) {
        items.forEach(itemText => {
            const li = document.createElement('li');
            // Basic check to prevent potential HTML injection if itemText somehow contains HTML
            li.textContent = typeof itemText === 'string' ? itemText : JSON.stringify(itemText);
            listElement.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.textContent = 'None found or analysis not applicable.';
        li.style.fontStyle = 'italic';
        li.style.color = '#888';
        listElement.appendChild(li);
    }
} 