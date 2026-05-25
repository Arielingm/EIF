
%% ========================================================
%  AUDITOR DE TAMAÑO Y NOMENCLATURA ESPECÍFICA
%  ========================================================
clear; clc;

input_folder = 'DataBase_original'; 
files = dir(fullfile(input_folder, '*.mat'));

% Reglas de validación
maquinas_validas = 'hebvpgorc'; 
controles_validos = 'lsd';

archivos_con_problemas = {};

fprintf('--- Iniciando Auditoría de Estructura ---\n');

for i = 1:length(files)
    name = lower(files(i).name);
    [~, fileName, ~] = fileparts(name); % Quitamos el .mat para contar bien
    
    problema_detectado = false;
    motivo = "";

    % 1. AUDITORÍA DE TAMAÑO (Regla: len > 4 o estructura extraña)
    % Si el nombre (sin extensión) tiene más caracteres de lo normal o letras extrañas
    if length(fileName) > 4
        problema_detectado = true;
        motivo = motivo + "Longitud excesiva (" + length(fileName) + " caracteres). ";
    end

    % 2. AUDITORÍA DE MAPEADO (Primeras 2 letras)
    m_letra = fileName(1);
    c_letra = fileName(2);
    
    if ~contains(maquinas_validas, m_letra)
        problema_detectado = true;
        motivo = motivo + "Máquina '" + m_letra + "' no mapeada. ";
    end
    
    if ~contains(controles_validos, c_letra)
        problema_detectado = true;
        motivo = motivo + "Control '" + c_letra + "' no mapeado. ";
    end

    % 3. REGISTRO DE ARCHIVOS "RAROS" (Como el bd54e.mat)
    if problema_detectado
        archivos_con_problemas{end+1} = sprintf('%s -> %s', name, motivo); %#ok<SAGROW>
    end
end

% --- REPORTE ---
if isempty(archivos_con_problemas)
    fprintf('\n✅ Todos los archivos tienen el tamaño y mapeado correcto.\n');
else
    fprintf('\n⚠️ SE ENCONTRARON %d ARCHIVOS FUERA DE REGLA:\n', length(archivos_con_problemas));
    fprintf('--------------------------------------------------\n');
    fprintf('%s\n', archivos_con_problemas{:});
    fprintf('--------------------------------------------------\n');
    
    % Guardar para revisión
    fid = fopen('errores_de_nombre.txt', 'w');
    fprintf(fid, 'REVISAR ESTOS ARCHIVOS:\n\n%s\n', archivos_con_problemas{:});
    fclose(fid);
end