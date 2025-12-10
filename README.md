# A Data Science Approach to Decoding Morse Code
![Morse Code DSP Banner](img/readme-header.png)

## Video Presentation
<iframe width="560" height="315" src="https://www.youtube.com/embed/8777BKS3vvc?si=LTRYt-d4ZPbcQQ_c" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### Introduction
**Project Scope:** This research covers a semester project in Data Science focusing on Morse code, a topic inspired by a summer robotics project.

**Relevance:** Audio and signal processing are an endless source of data, and signal processing more generally is of bedrock importance to modern life. Additionally, emergency response organizations like FEMA and the Red Cross also depend on basic radio to save lives during natural disasters.

**The Challenge:** Morse code is a human-centered protocol that is easy for people to learn but difficult for computers to decode because it lacks forward error correction and other features that ensure data integrity.

### Methodology and Tooling
**Critique of Standard Tools:** The researcher argues that common tools like Jupyter Notebooks are "antagonistic" to version control due to metadata that registers changes even when none are made. Raw shell scripts were also rejected because they lack the ability to "tell a story" with the data.

**Chosen Environment:**
    - Operating System: Linux was chosen for the freedom to customize software.
    - Version Control: A self-hosted, portable Git server was used to avoid the file size and type restrictions of platforms like GitHub.
    - The "Polyglot" Approach: The presenter utilized Emacs Org mode, which allows for running multiple programming languages (R, Elisp, Bash) seamlessly within the same file.
    - R Language: R was selected for its high-quality audio processing libraries that are backed by mathematical research papers.

### Exploratory Data Analysis (EDA)
**Corruption Checks:** The first step involved checking for file corruption by graphing the sample rate against the file length, which confirmed a linear pattern and indicated no major data loss.

**Coding Rate (WPM) Analysis:** By analyzing the coding rate (Words Per Minute), a bimodal distribution was observed in the dataset. This revealed two distinct groups: novices operating in the low-to-mid 20WPM range and experienced operators in the high 30WPM range.

**Edge Case Identification:** To test edge cases, the presenter used Root Mean Square (RMS) amplitude to isolate the cleanest (high RMS) and noisiest (low RMS) files for training data.

### Signal Processing and Results
**Clean vs. Noisy Data:**
    - Clean files showed clear amplitude spikes in the waveform, making them visually readable and decodable by standard algorithms.
    - Noisy files were visually indecipherable in the time domain due to high background noise masking the signal.
    - Spectrogram Analysis: The presenter applied a transform to view the noisy data as a spectrogram, which displays time, frequency, and amplitude (color) in three dimensions, making the Morse code visualizable again.
    - Filtering: A filter was developed to identify the specific signal frequency (approximately 600Hz) and isolate it. This transformed the noisy audio back into a clean waveform that is readable by traditional decoding methods.

### Conclusion
**Reproducibility:** Custom functions were "tangled" (extracted) from the Org mode file into a generic .R file, allowing the research to be easily reused or developed into an application later.

**Final Status:** Although time constraints prevented the completion of a fully automated decoding algorithm, the Fourier transforms and filtering methods were successful and deemed worth pursuing further.

### SCQA
1. Situation:
> The "State of the World": He establishes that audio and signal processing are "bedrock importance" to modern life, used by the FCC, FEMA, and in nature. He notes that Morse code is a "human protocol" that is highly effective for people to learn.
2. Complication:
>The Data Problem: While Morse code is easy for humans, it is "not friendly to computers to decode" because it lacks forward error correction and data integrity features.<br>
> The Tooling Problem: He discovered that standard data science tools like Jupyter Notebooks and RStudio were "antagonistic" to version control and independent research.
3. Question:
> Implicit Question 1: How can we effectively decode Morse code using data science techniques given these difficulties? (Addressed by asking "what to be done about that" regarding noisy files ). <br>
> Implicit Question 2: How can I conduct this research without the limitations of standard tools? (Addressed by his search for a solution that allows "maximum flexibility" without the opacity of raw shell scripts ).
4. Answer:
>The Workflow Solution: He details his "Polyglot" setup using Emacs Org mode, which allows him to run R, Bash, and Elisp in the same file, solving the version control and "storytelling" issues.<br>
>The Technical Solution: He demonstrates using spectrograms and custom filters to transform noisy, "unreadable" audio into clean waveforms that traditional algorithms can process.
5. Adherence to the "pyramid principle":
>The presentation follows the standard format of academic research: Hypothesis → Methodology → Experiment → Conclusion.<br>
>Barbara Minto explicitly argues against this format for business communication. She calls it the "Mystery Story" approach—where you force the audience to re-live your discovery process step-by-step. The Pyramid Principle demands you give the "solution to the mystery" first, and only use the steps to prove that the solution is valid.<br>
> _Note to self: Gemini cuts deep!_

# AI Disclaimer:
> Gemini was used for generating the above summary of my recorded video presentation, <br>
> as well as for evaluating my conformance to the Minto method (SCQA) prior to presenting in-person. <br>
> Likewise, the graphics for the video and the header of this file was created with Google Gemini's "NanoBanana" solution.
