import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

class AttendanceManager:
    def __init__(self, attendance_dir: Path):
        self.attendance_dir = Path(attendance_dir)
        self.attendance_dir.mkdir(parents=True, exist_ok=True)
    
    def mark_attendance(self, recognized_names: list):
        date = datetime.now().strftime("%Y-%m-%d")
        file_path = self.attendance_dir / f"attendance_{date}.csv"
        
        if file_path.exists():
            df = pd.read_csv(file_path)
        else:
            df = pd.DataFrame(columns=['Name', 'Time', 'Status'])
        
        current_time = datetime.now().strftime("%H:%M:%S")
        
        for name in recognized_names:
            if not df[(df['Name'] == name)].empty:
                continue
                
            new_row = {
                'Name': name,
                'Time': current_time,
                'Status': 'Present'
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        df.to_csv(file_path, index=False)
        logging.info(f"Attendance marked for {len(recognized_names)} students")