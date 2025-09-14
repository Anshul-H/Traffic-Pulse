FROM python:3.10-slim-bookworm



ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app


# Install small system dependencies you may need (git, ffmpeg for video processing, build tools)
RUN apt-get update \
&& apt-get install -y --no-install-recommends \
git \
ffmpeg \
build-essential \
&& rm -rf /var/lib/apt/lists/*


# Copy and install Python dependencies
COPY Requirements.txt ./Requirements.txt
RUN python -m pip install --upgrade pip \
&& pip install --no-cache-dir -r Requirements.txt


# Copy the app code
COPY . /app


# Set port for Streamlit (Hugging Face uses $PORT environment variable)
ENV PORT=7860
EXPOSE 7860


# Run the Streamlit app. Use streamlit_app.py as the entry point (adjust if different).
CMD ["bash", "-lc", "streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0"]
