-- Healthcare LLM Demo Database Setup
-- Creates the database, tables, and demo login accounts

-- Create and select database
CREATE DATABASE IF NOT EXISTS healthcarellm;
USE healthcarellm;

-- Drop dependent tables first
DROP TABLE IF EXISTS doctor_patient_access;
DROP TABLE IF EXISTS insurance_patient_access;
DROP TABLE IF EXISTS users;

-- Users table
-- Stores login accounts, password hashes, roles, and
-- patient bindings for patient users
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('patient', 'doctor', 'insurance') NOT NULL,
    patient_id VARCHAR(20) DEFAULT NULL,
    display_name VARCHAR(100) NOT NULL
);

-- Doctor-to-patient access mapping
-- Used to control which patients the doctor account can view
-- ---------------------------------------------------------
CREATE TABLE doctor_patient_access (
    id INT AUTO_INCREMENT PRIMARY KEY,
    doctor_id INT NOT NULL,
    patient_id VARCHAR(20) NOT NULL,
    CONSTRAINT fk_doctor_access_user
        FOREIGN KEY (doctor_id) REFERENCES users(id)
        ON DELETE CASCADE
);

-- Insurance-to-patient access mapping
-- Used to control which patients the insurance account can view
CREATE TABLE insurance_patient_access (
    id INT AUTO_INCREMENT PRIMARY KEY,
    insurance_id INT NOT NULL,
    patient_id VARCHAR(20) NOT NULL,
    CONSTRAINT fk_insurance_access_user
        FOREIGN KEY (insurance_id) REFERENCES users(id)
        ON DELETE CASCADE
);

-- Demo users
-- All demo accounts currently use the same password hash
-- for password: demo123
INSERT INTO users (username, password_hash, role, patient_id, display_name)
VALUES
('maria',      'scrypt:32768:8:1$R6CseyS3qo7cu0Lj$faf3d5229bd2446d5013dd782bd51ca52e137b6f7e27ca5c6dc32b6c5ae26157bd01d41e3334534ed9df18a7cd5f0281c40ef290942f8aa790bb26d264455885', 'patient',   'patient_001', 'Maria Santos'),
('james',      'scrypt:32768:8:1$R6CseyS3qo7cu0Lj$faf3d5229bd2446d5013dd782bd51ca52e137b6f7e27ca5c6dc32b6c5ae26157bd01d41e3334534ed9df18a7cd5f0281c40ef290942f8aa790bb26d264455885', 'patient',   'patient_002', 'James O''Brien'),
('priya',      'scrypt:32768:8:1$R6CseyS3qo7cu0Lj$faf3d5229bd2446d5013dd782bd51ca52e137b6f7e27ca5c6dc32b6c5ae26157bd01d41e3334534ed9df18a7cd5f0281c40ef290942f8aa790bb26d264455885', 'patient',   'patient_003', 'Priya Kapoor'),
('robert',     'scrypt:32768:8:1$R6CseyS3qo7cu0Lj$faf3d5229bd2446d5013dd782bd51ca52e137b6f7e27ca5c6dc32b6c5ae26157bd01d41e3334534ed9df18a7cd5f0281c40ef290942f8aa790bb26d264455885', 'patient',   'patient_004', 'Robert Chen'),
('doctor1',    'scrypt:32768:8:1$R6CseyS3qo7cu0Lj$faf3d5229bd2446d5013dd782bd51ca52e137b6f7e27ca5c6dc32b6c5ae26157bd01d41e3334534ed9df18a7cd5f0281c40ef290942f8aa790bb26d264455885', 'doctor',    NULL,          'Doctor Account'),
('insurance1', 'scrypt:32768:8:1$R6CseyS3qo7cu0Lj$faf3d5229bd2446d5013dd782bd51ca52e137b6f7e27ca5c6dc32b6c5ae26157bd01d41e3334534ed9df18a7cd5f0281c40ef290942f8aa790bb26d264455885', 'insurance', NULL,          'Insurance Account');

-- Doctor access: doctor1 can access all four patients
INSERT INTO doctor_patient_access (doctor_id, patient_id)
SELECT id, 'patient_001' FROM users WHERE username = 'doctor1';

INSERT INTO doctor_patient_access (doctor_id, patient_id)
SELECT id, 'patient_002' FROM users WHERE username = 'doctor1';

INSERT INTO doctor_patient_access (doctor_id, patient_id)
SELECT id, 'patient_003' FROM users WHERE username = 'doctor1';

INSERT INTO doctor_patient_access (doctor_id, patient_id)
SELECT id, 'patient_004' FROM users WHERE username = 'doctor1';

-- Insurance access: insurance1 can access all four patients
INSERT INTO insurance_patient_access (insurance_id, patient_id)
SELECT id, 'patient_001' FROM users WHERE username = 'insurance1';

INSERT INTO insurance_patient_access (insurance_id, patient_id)
SELECT id, 'patient_002' FROM users WHERE username = 'insurance1';

INSERT INTO insurance_patient_access (insurance_id, patient_id)
SELECT id, 'patient_003' FROM users WHERE username = 'insurance1';

INSERT INTO insurance_patient_access (insurance_id, patient_id)
SELECT id, 'patient_004' FROM users WHERE username = 'insurance1';

-- Optional verification queries
-- SELECT id, username, role, patient_id, display_name FROM users;
-- SELECT * FROM doctor_patient_access;
-- SELECT * FROM insurance_patient_access;
