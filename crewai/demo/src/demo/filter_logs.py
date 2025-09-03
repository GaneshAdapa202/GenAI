from datetime import datetime

def extract_logs_by_date(log_file, date_str):
    filtered_logs = []
    with open(log_file, "r") as f:
        for line in f:
            if line.startswith(date_str):
                filtered_logs.append(line.strip())
    return filtered_logs

def extract_logs_in_range(log_file, start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    filtered_logs = []

    with open(log_file, "r") as f:
        for line in f:
            try:
                log_date_str = line.split()[0]  # first part = YYYY-MM-DD
                log_date = datetime.strptime(log_date_str, "%Y-%m-%d")
                if start <= log_date <= end:
                    filtered_logs.append(line.strip())
            except Exception:
                continue
    return filtered_logs
