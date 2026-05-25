%% ========================================================
%  SISTEMA DE ESTRUCTURACIÓN INTEGRAL - VERSION ROBUSTA
%  ========================================================
clear; clc;

input_folder  = 'DataBase_original';
output_folder = 'Motor_DB/data';
index_folder  = 'Motor_DB/index';

if ~exist(output_folder, 'dir'), mkdir(output_folder); end
if ~exist(index_folder, 'dir'), mkdir(index_folder); end

%% ===================== MAPAS OFICIALES =====================

% Estado del generador (motor de carga)
g_map = containers.Map( ...
    {'h','e','t'}, ...
    {'GenHealthy','GenBrokenBar','GenNotHealthy'} );

% Estado máquina bajo test
m_valid = {'h','e','b','v','p','g','o','r','c'};

% Control
c_valid = {'l','s','d'};

% Velocidad
s_map = containers.Map( ...
    {1,2,3,5,6,7,9}, ...
    {'S1500','S1200','S900','S20','STrap','SSteps','SStart'} );

% Carga
l_map = containers.Map( ...
    {0,1,2,3,4,6,7,8}, ...
    {'L0','L25','L50','L75','L100','LVar0','LVar25','LVar50'} );

%% ===================== MASTER INDEX =====================

index_file = fullfile(index_folder, 'master_index.csv');

if ~exist(index_file, 'file')
    fid = fopen(index_file, 'w');
    fprintf(fid, ['Archivo_Nuevo,ID_Original,Generador,Maquina,Control,' ...
                  'Velocidad,Carga,Intento,Fecha\n']);
    fclose(fid);
end

fid = fopen(index_file, 'a');

files = dir(fullfile(input_folder, '*.mat'));

%% ===================== PROCESAMIENTO =====================

for i = 1:length(files)

    current_file = files(i).name;
    [~, nameOnly, ~] = fileparts(current_file);

    fprintf('\nProcesando: %s\n', current_file);

    data = load(fullfile(input_folder, current_file));

    % Extraer letras y números
    letter_groups = regexp(nameOnly, '[a-zA-Z]+', 'match');
    number_group  = regexp(nameOnly, '\d+', 'match', 'once');

    if isempty(letter_groups) || isempty(number_group)
        warning('Formato inválido: %s', current_file);
        continue;
    end

    prefix = lower(letter_groups{1});
    numbers = number_group;

    %% ===================== PREFIJOS =====================

    if length(prefix) == 2
        % Caso estándar: bs12
        gen_code = 'h';                 % Generador healthy por defecto
        m_code   = prefix(1);
        c_code   = prefix(2);

    elseif length(prefix) == 3
        % Casos especiales: ebd, tbd, thd, etc.
        gen_code = prefix(1);
        m_code   = prefix(2);
        c_code   = prefix(3);
    else
        warning('Prefijo inválido en %s', current_file);
        continue;
    end

    %% ===================== VALIDACIONES =====================

    % Generador
    if ~isKey(g_map, gen_code)
        warning('Generador no mapeado en %s', current_file);
        continue;
    end

    % Máquina
    if ~ismember(m_code, m_valid)
        warning('Máquina no válida en %s', current_file);
        continue;
    end

    % Control
    if ~ismember(c_code, c_valid)
        warning('Control no válido en %s', current_file);
        continue;
    end

    %% ===================== VELOCIDAD Y CARGA =====================

    if length(numbers) ~= 2
        warning('Error en velocidad/carga en %s', current_file);
        continue;
    end

    s_val = str2double(numbers(1));
    l_val = str2double(numbers(2));

    if ~isKey(s_map, s_val)
        warning('Velocidad no válida en %s', current_file);
        continue;
    end

    if ~isKey(l_map, l_val)
        warning('Carga no válida en %s', current_file);
        continue;
    end

    %% ===================== SUFIJO (TRIAL) =====================

    if length(letter_groups) > 1
        trial_id = letter_groups{end};  % b, c, e, s
    else
        trial_id = '1';
    end

    %% ===================== CREAR ESTRUCTURA =====================

    test = struct();

    test.meta.generator_code = gen_code;
    %test.meta.generator_desc = g_map(gen_code);

    test.meta.machine_code   = m_code;
    test.meta.control_code   = c_code;

    test.meta.speed_code = s_map(s_val);
    test.meta.load_code  = l_map(l_val);
    test.meta.trial      = trial_id;

    % Señales
    test.signals.electrical.u = data.iu;
    test.signals.electrical.v = data.iv;
    test.signals.electrical.w = data.iw;

    test.signals.vibration.front_DE = [data.AccDEY, data.AccDEZ];
    test.signals.vibration.rear_NDE  = [data.AccNDEY, data.AccNDEZ];

    test.signals.vibration.housing_uniaxial = data.a;
    test.signals.speed_raw = data.s;

    %% ===================== NUEVO NOMBRE =====================

    new_filename = sprintf('G%s_M%s_C%s_%s_%s_T%s.mat', ...
        gen_code, m_code, c_code, ...
        s_map(s_val), l_map(l_val), trial_id);

    save(fullfile(output_folder, new_filename), 'test', '-v7.3');

    %% ===================== REGISTRO =====================

    fprintf(fid, '%s,%s,%s,%s,%s,%s,%s,%s,%s\n', ...
        new_filename, current_file, ...
        gen_code, m_code, c_code, ...
        s_map(s_val), l_map(l_val), trial_id, datestr(now));

    fprintf('OK -> %s\n', new_filename);

end

fclose(fid);

disp('--- PROCESAMIENTO COMPLETADO CORRECTAMENTE ---');