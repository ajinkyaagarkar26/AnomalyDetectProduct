# Build stage
FROM python:3.10-slim as builder

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements/ /app/requirements/

# Install dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements/requirements.txt

# Runtime stage
FROM python:3.10-slim

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set work directory
WORKDIR /app

# Copy project files
COPY logdeep/ /app/logdeep/
COPY logparser/ /app/logparser/
COPY log_anomaly/ /app/log_anomaly/
COPY deeplog.py /app/
COPY data_process.py /app/
COPY configs/ /app/configs/
COPY utils/ /app/utils/
COPY model/deeplog/bestloss.pth /app/model/deeplog/bestloss.pth
COPY model/deeplog/scale.pkl /app/model/deeplog/scale.pkl
COPY model/vocab.pkl /app/model/vocab.pkl

# expose port
EXPOSE 8001

# run app
CMD ["uvicorn", "log_anomaly.app.main:app", "--host", "0.0.0.0", "--port", "8001"]