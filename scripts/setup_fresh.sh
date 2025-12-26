#!/bin/bash
set -e

echo "=========================================="
echo "Fresh Setup - Energy Forecasting Pipeline"
echo "=========================================="
echo ""

# Stop containers
echo "Stopping containers..."
docker-compose down

# Remove volumes
echo "Removing Docker volumes..."
docker volume rm load-forecasting-ml-pipeline_pgdata 2>/dev/null || echo "pgdata volume doesn't exist"
docker volume rm load-forecasting-ml-pipeline_mlflow-data 2>/dev/null || echo "mlflow-data volume doesn't exist"
docker volume rm load-forecasting-ml-pipeline_airflow-logs 2>/dev/null || echo "airflow-logs volume doesn't exist"
docker volume rm load-forecasting-ml-pipeline_airflow-home 2>/dev/null || echo "airflow-home volume doesn't exist"

# Remove local directories
echo "Removing local directories..."
sudo rm -rf artifacts/ models/ mlflow-data/ 2>/dev/null || echo "Directories already clean"

# Recreate directories
echo "Creating directories..."
mkdir -p artifacts models mlflow-data
chmod -R 777 artifacts/ models/ mlflow-data/

# Start services
echo ""
echo "Starting Docker services..."
source .env
docker-compose up -d

# Wait for PostgreSQL
echo ""
echo "Waiting for PostgreSQL to be ready..."
sleep 10
until docker exec energy-db pg_isready -U energy_user -d energy 2>/dev/null; do
    echo "Waiting for PostgreSQL..."
    sleep 2
done
echo "PostgreSQL is ready!"

# Verify airflow database exists (init-db.sh should have created it)
echo ""
echo "Verifying airflow database..."
if docker exec energy-db psql -U energy_user -d postgres -lqt | cut -d \| -f 1 | grep -qw airflow; then
    echo "✓ Airflow database exists"
else
    echo "✗ Airflow database missing - creating it manually"
    docker exec energy-db psql -U energy_user -d energy -c "CREATE DATABASE airflow;"
    docker exec energy-db psql -U energy_user -d energy -c "GRANT ALL PRIVILEGES ON DATABASE airflow TO energy_user;"
fi

# Initialize Airflow
echo ""
echo "Initializing Airflow database..."
docker exec energy-airflow-scheduler airflow db migrate

# Wait for webserver to be ready
echo ""
echo "Waiting for Airflow webserver to be ready..."
sleep 15
until docker exec energy-airflow-webserver curl -s http://localhost:8080/health 2>/dev/null | grep -q "healthy"; do
    echo "Waiting for webserver..."
    sleep 3
done
echo "Webserver is ready!"

# Create admin user
echo ""
echo "Creating Airflow admin user..."
docker exec energy-airflow-webserver airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com \
  --password admin

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next step: Load sample data"
echo ""
echo "  source .venv/bin/activate"
echo "  source .env"
echo "  python3 -m scripts.kaggle_to_db_df_mlf"
echo ""
echo "Access:"
echo "  - Airflow: http://localhost:8080 (admin/admin)"
echo "  - MLflow:  http://localhost:5001"
echo ""
