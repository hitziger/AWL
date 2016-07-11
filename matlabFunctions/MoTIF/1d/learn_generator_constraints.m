function dico = learn_generator_constraints(ref_signal, atom_size, nb_atoms_to_learn, filename)

% Parameters to be set
show_things     = 0;
init_dico_rand  = 1;

% Create initial dirac atom 
init_atom = zeros(atom_size,1);
init_atom(round(atom_size/2)+1,1) = 1;

fprintf('\tIterative learning\n');

% Get signal size
signal_size = size(ref_signal,1);

% Create base structure of the dictionary
dico.atom = zeros(atom_size,1);

% Take random positions for the patches patches

nb_patches = floor(signal_size/(2*atom_size))-2;
patch_positions = zeros(nb_patches,1);
for p = 1:nb_patches
    patch_positions(p) = (2*p + 1)*atom_size + 1;
end

fprintf('\tUsing %d patches\n',nb_patches);

% Create residual
residual = ref_signal;

max_pos   = zeros(nb_patches,1);
max_proj  = zeros(nb_patches,1);

% Create the constraint matrix
constraint_matrix = zeros(atom_size,atom_size);

% Recursive part
for a=1:nb_atoms_to_learn
    
    fprintf('\t\tLearning atom %d of %d\n',a,nb_atoms_to_learn);
    
    % Construct the matrix of the constraints
    if size(dico,2) > 1
        d = size(dico,2)
        tmp_dico = create_possible_atoms(dico(d).atom);
        shifted_atoms_found = zeros(atom_size, size(tmp_dico,2));
        for t=1:size(tmp_dico,2)
            shifted_atoms_found(:,t) = tmp_dico(t).atom;
        end
        
        constraint_matrix = constraint_matrix + shifted_atoms_found*shifted_atoms_found';
    end
    
    % Create random initial atom if needed
    if init_dico_rand
        init_atom = randn(size(init_atom));
        en = init_atom'*init_atom;
        init_atom = init_atom / sqrt(en);
        
%         if size(dico,2) > 1
%             
%             init_atom  = zeros(size(init_atom));
%             worst_proj = 1000000000;
%             
%             for ra = 1:20000
%                 loc_atom = randn(size(init_atom));
%                 en = loc_atom'*loc_atom;
%                 loc_atom = loc_atom / sqrt(en);
%                 proj = 0;
%                 for d=2:size(dico,2)
%                     proj = proj + abs( loc_atom'* constraint_matrix*loc_atom);
%                 end
%                 
%                 if proj < worst_proj
%                     worst_proj = proj;
%                     init_atom = loc_atom;
%                 end
%             end
%         end
    end
%     
%     if a>1
%         init_atom = dico(a).atom;
%     end
     
    % FFT filtering of the residual
    filtered_residual = fftfilt(flipud(init_atom),residual);

    % Find maximum on all patches
    for p=1:nb_patches   
        local_patch_pos = patch_positions(p);
        [max_proj(p), max_pos(p)] = max(abs(filtered_residual(round(local_patch_pos+atom_size/2):round(local_patch_pos+atom_size/2-1+atom_size))));
    end

    % Where take the patch so that it is centered
    patch_positions_center = patch_positions -round(atom_size/2) + max_pos;
    
    % The iterations will be interupted depending on a criterion tested at
    % the end of the while.
    more_iterations = 1;
    
    % Begin iteration
    iter_dico(1).atom = init_atom;
    iter_dico(1).iter_patch_positions_center = patch_positions_center;
    iter_counter = 1;   
    
    
    while more_iterations
        
        % Increment counter
        iter_counter = iter_counter + 1;
        
        % Modify the positions to center the maximum energy in the middle
        % of the signal
        past_atom = iter_dico(iter_counter-1).atom;
        diff_to_center = round(mean(([1:atom_size]-atom_size/2)*(past_atom.*past_atom)));
        
        if abs(diff_to_center<5)
            diff_to_center = 0;
        end
    
        % Create matrix with all centered patches
        patch_matrix_center = zeros(atom_size, nb_patches);
        for p = 1:nb_patches
            
            %max_proj(p)

            
            py = iter_dico(iter_counter-1).iter_patch_positions_center(p)+diff_to_center; 
            
            if py + atom_size -1 >= max(size(residual)) || py < 1
                py = round(iter_dico(iter_counter-1).iter_patch_positions_center(p));
            end
            
            test = residual(py:py+atom_size-1, 1);
            
            
%             en = test'*test;
%             if en ~= 0
%                 test = test./sqrt(test'*test);
%             end
%             
%             test = test*max_proj(p);
            
            patch_matrix_center(:,p) = test;
           

        end
        
        % No display during eigenvector search
        opts = struct('disp',{});
        opts(1).disp=0;
        
        opt.MAXEIG = 1;
        opt.DISP = 0;
        
        if size(dico,2) > 1
            
            % Generalized eigenvalue problem
            [new_atom,D] = eigs(patch_matrix_center*patch_matrix_center',constraint_matrix,1,'lm',opts);
% 
%             mval = 0;
%             mb=0;
%             for i=1:200
%                 gf = new_atom(:,i);
%                 gf = gf./sqrt(gf'*gf);
%                 new_atom(:,i) = gf;
%                 
%                 val = gf'*constraint_matrix*gf;
%                 if val > mval
%                     mval = val;
%                     mb = i;
%                 end
% 
%             end
%             fo = 0;
%             mval
%             new_atom = new_atom(:,mb);
%             plot(new_atom);
%             [new_atom2,D] = eigs(patch_matrix_center*patch_matrix_center',1,'lm',opts);
%             plot(new_atom2);
            %tot_en = sum(abs((new_atom'*patch_matrix_center)));
            %fprintf('\t\t\tMean energy is %f\n',tot_en);

        else
            % Take the largest eigenvector
            [new_atom,D] = eigs(patch_matrix_center*patch_matrix_center',1,'lm',opts);
            
            %tot_en = sum(abs((new_atom'*patch_matrix_center)));
            %fprintf('\t\t\tMean energy is %f\n',tot_en);
            
        end
        
        en = new_atom' * new_atom;
        new_atom = new_atom/sqrt(en);
        
        if sum(sum(new_atom .* iter_dico(iter_counter-1).atom)) < 0
            new_atom = - new_atom;
        end
        
        % Add new atom to history
        iter_dico(iter_counter).atom = new_atom;
        
        % Recompute the positions with new atom
        max_pos_iter   = zeros(nb_patches,1);
        
        % Filter the signal with actual gen function
        filtered_residual = fftfilt(flipud(iter_dico(iter_counter).atom),residual);
        

        for p=1:nb_patches

            py = patch_positions(p);
            [max_proj(p), max_pos_iter(p)] = max(abs(filtered_residual(round(py+atom_size/2):round(py+atom_size/2-1+atom_size))));

        end
        
        patch_positions_center = patch_positions - round(atom_size/2) + max_pos_iter;
        iter_dico(iter_counter).iter_patch_positions_center = patch_positions_center;
        
        % Compare the positions to know if it is worth to make more
        % iterations
        delta = mean(abs(iter_dico(iter_counter).iter_patch_positions_center-iter_dico(iter_counter-1).iter_patch_positions_center));       
        l2atom = sqrt(sum((iter_dico(iter_counter).atom - iter_dico(iter_counter-1).atom).^2));
        
        fprintf('\t\t\t\tIter %d: delta %f and L2 of diff %f\n',iter_counter,delta,l2atom);
        
        if delta < 0.01 || iter_counter > 100 || l2atom < 0.000000001
            more_iterations = 0;
        end
    end
    
    % Add found atom to the dictionary
    dico(size(dico,2)+1).atom = iter_dico(iter_counter).atom;

    % Save the current state of the dictionary to a file
    save(filename, 'dico');
    
    if show_things
        
        figure(100)
        for toto = 1:iter_counter
            subplot(iter_counter,1,toto)
            plot( iter_dico(toto).atom )
        end
        
        
%         for i = 1:iter_counter
%             figure(100+i)
%             af = abs(fft(iter_dico(i).atom));
%             af_s = af(1:max(size(iter_dico(i).atom)/2));
%     
%             subplot(2,1,1)
%             plot(iter_dico(i).atom)
%             title(strcat('Iteration :  ',num2str(i)));
%     
%             subplot(2,1,2)
%             plot(af_s)
%         end
%         
%         if a>1
%         
%             % Compute correlation with previous atom
%             % 1 - start of previous with end of current
%             ap_full = dico(size(dico,2)-1).atom;
%             ac_full = dico(size(dico,2)).atom;
%             sc_vect = zeros(2*atom_size-2,1);
%             for c=1:atom_size-1
%                 start_prev = 1;
%                 start_curr = atom_size-c;
%                 stop_prev  = 1+c;
%                 stop_curr  = atom_size;
% 
%                 ap_wind = ap_full(start_prev:stop_prev);
%                 ap_wind = ap_wind / sqrt(ap_wind'*ap_wind);
% 
%                 ac_wind = ac_full(start_curr:stop_curr);
%                 ac_wind = ac_wind / sqrt(ac_wind'*ac_wind);
% 
%                 sc = ap_wind' * ac_wind;
%                 sc_vect(c) = sc;
%             end
%             % 1 - end of previous with start of current
%             for c=0:atom_size-1
%                 start_prev = 1+c;
%                 start_curr = 1;
%                 stop_prev  = atom_size;
%                 stop_curr  = atom_size-c;
% 
%                 ap_wind = ap_full(start_prev:stop_prev);
%                 ap_wind = ap_wind / sqrt(ap_wind'*ap_wind);
% 
%                 ac_wind = ac_full(start_curr:stop_curr);
%                 ac_wind = ac_wind / sqrt(ac_wind'*ac_wind);
% 
%                 sc = ap_wind' * ac_wind;
%                 sc_vect(atom_size+c) = sc;
%             end
%             figure(150)
%             subplot(2,1,1)
%             plot(sc_vect)
%             subplot(2,1,2)
% 
%             ap_full_f = abs(fft(ap_full));
%             ap_full_f_s = ap_full_f(1:max(size(ap_full)/2));
% 
%             ac_full_f = abs(fft(ac_full));
%             ac_full_f_s = ac_full_f(1:max(size(ac_full)/2));
% 
%             plot(ap_full_f_s)
%             hold on
%             plot(ac_full_f_s, 'red')
%             hold off
%         end
        pause;
    end
        
    % Filter the signal with actual gen function
%     loc_centroid = tree_get_centroid(dico, size(dico,2), atom_size/2);
%     filtered_residual = fftfilt(flipud(loc_centroid) ,residual);
%     filtered_residual_abs = abs(filtered_residual);
% 
%     for p=1:nb_patches
% 
%         py = patch_positions(p);
%         [max_val(p), max_pos(p)] = max(filtered_residual_abs(py+atom_size/2:py+atom_size/2-1+atom_size));
% 
%     end
%     patch_positions_center = patch_positions - atom_size/2 + max_pos;
%     
%     % At all previous positions, take this atom out of the signal
%     for p=1:nb_patches
%         
%         %figure(33)
%         
%         py = patch_positions_center(p);
%         res = residual(py:py+atom_size-1, 1);
%         %subplot(4,1,1)
%         %plot(loc_centroid)
%         %subplot(4,1,2)
%         %plot(res)
%         
%         proj = loc_centroid' * res;
%         res = res - proj*loc_centroid;
%         %subplot(4,1,3)
%         %plot(res)
%         %subplot(4,1,4)
%         %plot(residual(patch_positions(p):patch_positions(p)+atom_size-1))
%         residual(py:py+atom_size-1, 1) = res;
%      
%         %max_pos(p)
%         
%         %pause;
%     end
end        