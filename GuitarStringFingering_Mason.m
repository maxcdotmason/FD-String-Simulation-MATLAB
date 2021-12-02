%   Max Mason
%   GuitarStringFingering_Mason.m

%   9 strings are simulated corresponding to the six strings of a
%   classical guitar, the D string on a violin, the D string on a
%   electric bass and the B string on an electric guitar. 

%   A method for simple simulation of fretting is presented involving the
%   time varient setting of simulated points to zero displacement. This was
%   coupled with a simple spreading function and a raised cosine
%   application for discontinuity avoidance. This method introduces error
%   in fret tuning due to the limited spacial fidelity which is explored. 

%   seperately a method for implementing multiple excitations is modelled
%   based on the simple pluck or strike. The seperation of this from the
%   fretting allows for the fretting and plucking to be controlled
%   individually which allows for a realistic rendition of an original
%   composition to be performed with hammer on/offs. The fretting model
%   also means that the 'finger' is removed at realistic points which the
%   careful ear can pick out, especially if a single string is listened to
%   (console command "soundsc(y(:,2),SR)").

%   in adition, a mechanism for slides along the fretboard is implemented
%   which linearly moves the 'finger' between frets over a defined period. 
%   this system uses only the existing points and no interpolation is 
%   performed. 

%   for excitation of the violin model a simple bow model is implemented as
%   found in NSS. 

%   references can be found at the end of the code.

clear all
close all

%%%%% flags %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

itype = 2;                  % type of input: 1: pluck, 2: struck
writeOut = 0;               % write data to wav file? 1 = yes, 0 = no

%%%%% parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Bass string parameters
Bassr = 0.0008255;
BassE = 206E6;
BassStiffness = 0.25*pi*Bassr^4*BassE;

% Length of each string
ScaleLength = [0.65,    0.65,   0.65,   0.65,   0.65,   0.65,   0.8634, 0.33,   0.65];

% individual string parameters

%           e           B           G           D           A           E           Bass            Violin      Solo            
T =         [70.3,      53.4,       58.3,       71.2,       73.9,       71.6,       51.2,           51,         58.8];                  % tension (N) https://www.gamutmusic.com/violin-equal-tensioned
r =         [0.000362,  0.000415,   0.000512,   0.000381,   0.00046,    0.000559,   0.0008255,      0.00061,    0.000215];                 % string radius (m)
stiffness = [0.00013,   0.00016,    0.00031,    0.000051,   0.00004,    0.000057,   BassStiffness,  0.00061359, 0.00032];         % equivelent to IE: http://www2.eng.cam.ac.uk/~jw12/JW%20PDFs/Guitar_II.pdf
rho =       [923.287,   959.733,    1056.62,    4275.97,    5497.24,    6360.95,    7860,           4800,       3959.733];                 % density (kg/m^3)
T60 =       [4.6,       5.75,       4.06,       5.75,       7.66,       5.75,       100,            10,         10];                    % T60 (s)     http://knutsacoustics.com/files/Damping.pdf
sig1 =      [0.0001,    0.0001,     0.0001,     0.0001,     0.0001,     0.0001,     0.005,          0.00001,    0.0001];


isBowed =       [0 0 0 0 0 0 0 1 0];            % String is to be excited with a bow rather than a strike

%%%%% Score %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tempo = 66;                                     % tempo of peice in BPM
z = 60/(tempo*3);                               % base unit of time for piece

%   Start time of fretting for each note for each string
%           note 1  2       3       4       5       6       7
Note_st =  [18.5*z, 12*z,   0,      0,      0,      0,      0;      % string 1
            12*z,   42.5*z, 0,      0,      0,      0,      0;      % 2
            12*z,   24*z,   36*z,   48*z,   0,      0,      0;      % 3
            24*z,   36*z,   0,      0,      0,      0,      0;      % 4
            24*z,   48*z,   0,      0,      0,      0,      0;      % 5
            24*z,   0,      0,      0,      0,      0,      0;      % 6
            24*z,   36*z,   48*z,   0,      0,      0,      0;      % 7
            0,      24*z,   36*z,   48*z,   0,      0,      0;      % 8
            12*z,   21*z,   24*z,   33*z,   36*z,   45*z,   48*z]'; % 9
        
%           Sequence of frets for each note for each string 
%           (fret number +1 as open string included)
NoteSeq =  [3,  0,  0,  0,  0,  0,  0;
            4,  3,  0,  0,  0,  0,  0;
            3,  4,  3,  2,  0,  0,  0;
            5,  3,  0,  0,  0,  0,  0;
            5,  3,  0,  0,  0,  0,  0;
            3,  0,  0,  0,  0,  0,  0;
            5,  8,  3,  0,  0,  0,  0;
            12, 5,  8,  3,  0,  0,  0;
            11, 13, 8,  11, 4,  3,  6]';

%           duration each note should be fretted for
Note_Dur = [0.5*z,  0,      0,      0,      0,      0,      0;
            12*z,   0.5*z,  0,      0,      0,      0,      0;
            12*z,   12*z,   12*z,   12*z,   0,      0,      0;
            12*z,   24*z,   0,      0,      0,      0,      0;
            12*z,   12*z,   0,      0,      0,      0,      0;
            12*z,   0,      0,      0,      0,      0,      0;
            12*z,   12*z,   12*z,   0,      0,      0,      0;
            24*z,   12*z,   12*z,   30*z,   0,      0,      0;
            9*z,    3*z,    9*z,    3*z,    9*z,    3*z,    12*z]';

%           Number of fretted notes for each string        
NumNotes = [1,2,4,2,2,1,3,4,7];

%           Is each note a slide from the previous note?
slide =    [0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0;
            0,1,1,1,0,0,0,0;
            0,1,0,1,0,0,0,0]';

%           time of each pluck for each string        
exc_st =   [z*17,   z*18,   51.5*z, 57.5*z, 0,      0,      0,      0;
            z*15,   z*20,   z*27,   z*33,   41*z,   42*z,   51.4*z, 57.4*z;
            z*14,   z*21,   z*29,   z*34,   39*z,   44*z,   51.3*z, 57.3*z;
            12*z,   z*35,   38*z,   45*z,   51.2*z, 56*z,   57.2*z, 0;
            z*26,   z*32,   36*z,   50*z,   51.1*z, 55*z,   57.1*z, 0;
            z*24,   z*30,   48*z,   51*z,   54*z,   57*z,   0,      0;
            12*z,   24*z,   36*z,   48*z,   0,      0,      0,      0;
            0,      0,      0,      0,      0,      0,      0,      0;
            12*z,   21*z,   24*z,   33*z,   36*z,   45*z,   48*z,   0];

%           number of plucks for each sting        
NumStrikes = [4,8,8,7,7,6,4,0,7];

%%%%% I/O %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SR = 96000;                         % sample rate (Hz)
Tf = 20;                            % duration of simulation (s)

xi = 0.9;                           % coordinate of excitation (normalised, 0-1)
famp = 1;                           % peak amplitude of excitation (N)
dur = 0.001;                        % duration of excitation (s)
slide_dur = 0.2;                    % duration of slides
xo = 0.75;                          % coordinate of output (normalised, 0-1)
TransDur = 0.01;                    % duration of transition between finger fully off and fully on

ayy = 10;                           % Bow model parameter
FB = 1;                             % bow force
Vb = 0.2;                           % bow speed

%%%%%%%%%%%%%%%% NO USER DEFINED INPUTS BELOW THIS POINT %%%%%%%%%%%%%%%%%%

%%%%% Input checking %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

assert(all(T > 0),'Tension must be positive, stopping script')
assert(all(r > 0),'String radius must be positive, stopping script')
assert(all(rho > 0),'Density must be positive, stopping script')
assert(all(T60 > 0),'T60 time must be positive, stopping script')
assert(all(SR > 0),'Sample rate must be positive, stopping script')
assert(all(Tf > 0),'Simulation time must be positive, stopping script')
assert((0 < xi)&&(xi < 1),'pluck position must be between 0 and 1, stopping script')
assert(dur > 0,'Excitation duration must be positive, stopping script')
assert((0 < xo)&&(xo < 1),'output position must be between 0 and 1, stopping script')
assert(all(ScaleLength > 0),'String length must be positive, stopping script')

%%%%% derived parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Nf = floor(SR*Tf);                  % number of time steps

Area = pi*r.^2;                     % string cross-sectional area

c = sqrt(T./(rho.*Area));           % wave speed
K = sqrt(stiffness./(rho.*Area));   % stiffness constant
sig = 6*log(10)./T60;               % frequency independent loss parameter

Length = zeros(13,9);               % Length of string at each fret
Length(1,:) = ScaleLength;          % open strings equal in length to scale length

%%%%%% grid and fret positioning %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   for each fret, calculate length
for fretnum = (2:13)                
    Length(fretnum,:) = Length(fretnum-1,:)-Length(fretnum-1,:)/17.817;
end

k = 1/SR;                           % time step

%   hmin from stability conditions
hmin = sqrt(0.5*((c.^2.*k.^2)+sqrt(c.^4*k.^4 + 16*K.^2.*k.^2)));
N = floor(ScaleLength./hmin);       % set integer number of segments
h = ScaleLength./N;                 % reset grid spacing

%   location of each fret in spacial samples
FretLocation = N - round((Length./ScaleLength).*N);

%   Error in fret location
FretDiscrepancy = ((round(Length.*N) - Length.*N).*h*1000)';
    
%   Plot error of each fret, this error leads to tuning issues but can be
%   improved with increased sample rate.

figure(1)
bar(FretDiscrepancy)
xlabel('String/Fret')
ylabel('Discrepancy of fret position from ideal (mm)')
title('Change to fret positions due to spacial fidelity')
drawnow

assert(all(N > 0),'current paramters lead to more then 10000 spacial points, consider reducing the sample rate or length')

%%%%% fretting timing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   start, transition and end timings for all frettings
NoteStart = floor(Note_st*SR)+1;
NoteTransTime = (0:k:TransDur)/TransDur;
NoteTransEnd = floor(NoteStart + TransDur*SR);
NoteDurEnd = floor(NoteTransEnd + Note_Dur*SR);
NoteEnd = floor(NoteDurEnd + TransDur*SR);

%%%%% for each string %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y = zeros(Nf,9);                    % initialise output

for str = (1:9)                         
    
    f = zeros(Nf,1);                    % initialise empty excitation vector
    Q = zeros(Nf,1);                    % initialise empty fret timing vector
    P = zeros(N(str)-1,1);              % initialise empty current fret position vector
    
    for a = (1:NumStrikes(str))             % for each excitation
        
        %   find start and end times and time vector
        fstart = floor(exc_st(str,a)*SR)+1; 
        fend = floor(fstart + dur*SR);
        ftime = (0:k:dur)/dur;
        
        %   populate excitation vector 
        f(fstart:fend) = 0.5*famp*(1 - cos(itype*pi()*ftime));
        
    end
    
%%%%%%%%% scheme coefficients %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    e = ones(N(str)-1,1);               % diagonal length of ones 
    
    %   second spacial derivative
    Dxx = (1/h(str)^2)*spdiags([e -2*e e], -1:1, N(str)-1,N(str)-1);
    
    %   fourth spacial derivative, including simply supported end condition
    Dxxxx = (1/h(str)^4)*spdiags([e -4*e 6*e -4*e e], -2:2, N(str)-1,N(str)-1);
    Dxxxx(1,1) = 5/h(str)^4;
    Dxxxx(N(str)-1,N(str)-1) = 5/h(str)^4;
    
    %   scheme coefficients
    A = (1 + sig(str)*k)*speye(N(str)-1) - sig1(str)*k*Dxx;
    B =  -2*speye(N(str)-1)-((c(str)^2)*(k^2)*Dxx)+((K(str)^2)*(k^2)*Dxxxx);
    C = speye(N(str)-1)*(1-sig(str)*k)+sig1(str)*k*Dxx;
    
    %   input vector
    J = zeros(N(str)-1,1);
    J(round(xi*N(str)-1)) = 1*(k^2)/(rho(str)*Area(str)*h(str));
    
    %   output vector
    cOut = zeros(N(str)-1,1);
    cOut(round(xo*N(str)-1)) = 1;
    
%%%%%%%%% Fret timing vector %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if NumNotes(str) ~= 0
        for b = (1:NumNotes(str))           % for each note
            
            if slide(b+1,str) == 1              % if note slides into next
                
                %   note starts with a cosine then remains high until the end of the next note
                Q(NoteStart(b,str):NoteTransEnd(b,str)) = (0.5*(1 - cos(pi*NoteTransTime)));
                Q(NoteTransEnd(b,str):NoteDurEnd(b+1,str)) = ones(NoteDurEnd(b+1,str)-NoteTransEnd(b,str)+1,1);
                Q(NoteDurEnd(b+1,str):NoteEnd(b+1,str)) = (0.5*(1 - cos(pi*NoteTransTime+pi)));
                                                
            else                                % if the note does not slide into the next
                
                if slide(b,str) == 1                % if this note was slid to, do nothing
                else
                    
                %   note starts with cosine then ends with cosine    
                Q(NoteStart(b,str):NoteTransEnd(b,str)) = (0.5*(1 - cos(pi*NoteTransTime)));
                Q(NoteTransEnd(b,str):NoteDurEnd(b,str)) = ones(NoteDurEnd(b,str)-NoteTransEnd(b,str)+1,1);
                Q(NoteDurEnd(b,str):NoteEnd(b,str)) = (0.5*(1 - cos(pi*NoteTransTime+pi)));
                
                end
            end
        end
        
%%%%%%%%%%%%% Fret current position vector %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %   spreading function centered on first note fret position 
        P(FretLocation(NoteSeq(1,str),str)) = 1;
        P(FretLocation(NoteSeq(1,str),str)+1) = 0.5;
        P(FretLocation(NoteSeq(1,str),str)-1) = 0.5;
        
    end
    
%%%%% initialise scheme variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    u2 = zeros(N(str)-1,1);             % state
    u1 = u2;                            % state
    u = u2;                             % state
    
    d = 1;                              % current note number
    timer = 0;                          % slide timer
    neta = 0;                           % initial friction coefficient
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for n=1:Nf %%%%% main loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

        %%%%% update state %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if isBowed(str) == 1                    % if string is excited by a bow
            
            %   update state
            u = A\((1-P*Q(n)).*(-B*u1-C*u2+FB*J*sqrt(2*ayy)*neta*exp(-ayy*neta^2 + 0.5)));
            %   update bow friction
            neta = (u(round(xi*N(str)-1))-u1(round(xi*N(str)-1)))/k - Vb;
            
        else                                    % if the string is plucked
            
            %   update state
            u = A\((1-P*Q(n)).*(-B*u1-C*u2+J*f(n)));
            
        end
        
        %%%%% update fret position %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if d <= NumNotes(str)                   % if there is a note to be played
            
            if slide(d,str) == 1                    % if the current note is slid to
                
                if timer < slide_dur                    % if the slide is not complete
                    
                    timer = timer + k;                  % increment timer
                    
                    %   current 'finger' position
                    Loc = ((FretLocation(NoteSeq(d,str),str) - FretLocation(NoteSeq(d-1,str),str))*timer/slide_dur)+FretLocation(NoteSeq(d-1,str),str);
                    
                    %   update Fret current position vector
                    P = zeros(N(str)-1,1);
                    P(round(Loc)) = 1;
                    P(round(Loc)+1) = 0.5;
                    P(round(Loc)-1) = 0.5;
                    
                else                                    % if slide is complete
                    
                    %   if it's time to go to the next note
                    if n > NoteEnd(d,str) && d < NumNotes(str)
                        
                        d = d + 1;                          % increment note counter
                        
                        %   update Fret current position vector
                        P = zeros(N(str)-1,1);
                        P(FretLocation(NoteSeq(d,str),str)) = 1;
                        P(FretLocation(NoteSeq(d,str),str)+1) = 0.5;
                        P(FretLocation(NoteSeq(d,str),str)-1) = 0.5;
                        
                        timer = 0;                          % reset slider timer
                        
                    end                                      
                end
            end
            
            if d < NumNotes(str)                    %   if there is a note to go to & the current note is not slid to
                if n > NoteEnd(d,str)                   % if it's time to move to the next note
                    
                    d = d + 1;                          % increment note counter
                    
                    %   update Fret current position vector
                    P = zeros(N(str)-1,1);
                    P(FretLocation(NoteSeq(d,str),str)) = 1;
                    P(FretLocation(NoteSeq(d,str),str)+1) = 0.5;
                    P(FretLocation(NoteSeq(d,str),str)-1) = 0.5;
                    
                end
            end
        end
        
        %%%%% Read Output %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %   differentiate and save output
            y(n,str) = (cOut'*u-cOut'*u1)/(k);
           
        %%%%% shift state %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        u2 = u1;
        u1 = u;
        
    end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end main loop %%%%%%%%%%%%%%%%%%%
    
    %   inform user of progress
    Message = [' String ',num2str(str),' complete'];
    disp(Message)
    
end                                 % end loop for each string

%%%%% play sound %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   read in impulse response
ImpulseResponceFileName = "Holy Grail SR25 SB3 (blend).wav";
[Reverb, ReverbFs] = audioread(ImpulseResponceFileName);

%   mix and pan output
yl = sum(y(:,(1:6)),2) + y(:,7)*10 + y(:,8)*0.075 + y(:,9)*1;
yr = sum(y(:,(1:6)),2) + y(:,7)*15 + y(:,8)*0.05 + y(:,9)*0.75;

%   convolve with reverb
yr = conv(yr,Reverb);
yl = conv(yl,Reverb);

%   create sound output
yOut = [yl,yr];

yOut = 0.9*yOut/max(abs(yOut(:)));

%   play sound
soundsc(yOut,SR);

%   save output
if writeOut == 1
    filename = '20sClip_S1832740_Mason.wav';
    audiowrite(filename,yOut,SR);
end

%%%%% Discussion %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   This simple method of fret simulation is remarkably good at achieving
%   the correct tone for a guitar or bass simulation. The violin does not
%   work quite so well, purhaps because the violin does not have frets and 
%   therefore the spreading term need be much smoother and wider. The 
%   violin has a very different tone when played open string. 

%   The tuning issues presented by the fact the frets can only be simulated
%   at calculated points is noticable, especially on the higher strings
%   which have much reduced spacial fidelity due to the higher wave speed
%   but further improvement could be acheived with an even higher sample
%   rate. 

%   The default sample rate (96kHz) does lead to a long computation time
%   although, as the strings are currently uncoupled and in seperated for
%   loops, there is scope for multi core processing which would reduce this
%   time greatly. 

%   An alternative to this would be to combine the strings into one matrix
%   which would also open the door to the possibility of coupling the
%   strings to provide a degree of autoexcitation. 

%   The slide mechanism also works well, my ear cannot pick out any jumps
%   between points during slides, indeed on a guitar the frets mean these
%   jumps should be even larger. It is a shame that the new 'finger'
%   position is cureently calculated in the processing loop as this will
%   effect efficiency.

%   
%%%%% References %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   D’Addario Pro Arte´ “Composites, hard tension” guitar strings
%   variables found in and from D'Addario: 
%   Plucked Guitar Transients: Comparison of Measurements and Synthesis,
%   J. Woodhouse Cambridge University Engineering Department, Trumpington St, Cambridge CB2 1PZ, UK.
%   http://www2.eng.cam.ac.uk/~jw12/JW%20PDFs/Guitar_II.pdf

%   damping coeficients converted to T60 according to knutacoustics.com
%   http://knutsacoustics.com/files/Damping.pdf

%   bass string parameters collected from:
%   https://i.imgur.com/3gJo2wa.jpg
%   and
%   On inharmonicity in bass guitar strings with application to tapered
%   and lumped constructions, Jonathan A. Kemp, SN Applied Sciences (2020) 2:636 | https://doi.org/10.1007/s42452-020-2391-2
%   https://link.springer.com/content/pdf/10.1007/s42452-020-2391-2.pdf

%   violin parameters found at:
%   https://www.gamutmusic.com/violin-equal-tensioned
%   or otherwise estimated

%   Electric guitar b string paramters extrapolated from various values, no
%   real string modelled





