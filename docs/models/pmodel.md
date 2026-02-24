# PModel (Aedes Albopictus)

## Model Overview

---

## How to run the model?

1. Locate the file `src/heiplanet_models/global_settings.yaml`.

2. Modify the path where you have the data, for instance:
```yaml
ingestion:
    path_root_datasets: "data/in/"
```
3. Run the Python script `src/heiplanet_models/Pmodel/run_model.py`
```bash
python run_model.py
```

---

## Architecture

---

### Inputs
| Dataset                  | Description              | Format  |
| ------------------------ | ------------------------ | ------- |
| ERA5Land Temperature     | Gridded temperature data | NetCDF4 |
| ERA5Land Precipitation   | Gridded precipitation    | NetCDF4 |
| Human population         | Population distribution  | NetCDF4 |
| Initial conditions (opt) | Starting model state     | NetCDF4 |


### Outputs
| Dataset            | Description                                                | Format  |
| ------------------ | ---------------------------------------------------------- | ------- |
| Mosquito Abundance | Gridded mosquito abundance for the Aedes Albopictus specie | NetCDF4 |


---

### Model Structure
#### Equations

#### Workflow

---
### Examples

---

## References:
- Barman, S., Semenza, J.C., Singh, P. et al. A climate and population dependent diffusion model forecasts the spread of Aedes Albopictus mosquitoes in Europe. Commun Earth Environ 6, 276 (2025). [https://doi.org/10.1038/s43247-025-02199-z](https://doi.org/10.1038/s43247-025-02199-z)
- [List of equations]()
- [Supplementary material](https://static-content.springer.com/esm/art%3A10.1038%2Fs43247-025-02199-z/MediaObjects/43247_2025_2199_MOESM2_ESM.pdf)

## Authors & Contact




---



## [deprecated] How the original Octave/Matlab code works

### Entry point name: `aedes_albopictus_model.m`

1. Create variables to assign the prefix of each dataset that will be used
   ```matlab
   tmean_prefix = 'ERA5land_global_t2m_daily_0.5_';
   pr_prefix = 'ERA5land_global_tp_daily_0.5_';
   dens_prefix = 'pop_dens_';
   ```

2. Create a `for` loop to iterate over years. The following operations are done in each loop:
    
    a. The name for each dataset is completed with the year and extension:
    ```matlab
    tmean = [tmean_prefix, num2str(years), '.nc'];
    pr = [pr_prefix, num2str(years), '.nc'];
    dens = [dens_prefix, num2str(years), '_global_0.5.nc'];
    ```
    
    b. Log the year that is being processed:
    ```matlab
    year = years; % change years according to file name
    disp(['Year in Process: ', num2str(years)]);
    ```

    c. Run the model that calculates the differential equations. This model receives the following datasets and variable:

    - dataset `tmean`: mean temperature dataset
    - dataset `pr`: precipitation dataset
    - dataset `dens`: population density dataset
    - variable `year`: year of the dataset
    
    ```matlab
    %model_run(tmax,tmin,tmean, pr,dens,year);
    model_run(tmean, pr,dens,year);            % for tmean only with NO DTR (Tmax or Tmin is not present)
    disp(['Year done: ', num2str(years)]);
    ```  
3. End the program

### Name: `model_run.m`
1. Create a variable name for the output file.
```matlab
outfile = strcat('Mosquito_abundance_Global_', num2str(year), '.nc')      % change the output file name as required

if exist(outfile) == 2
    delete(outfile)
end
```
2. Copy the dataset that contains the precipitation (one of the ERA5*.nc) and store this copy using the variable name for the output file.
```matlab
copyfile(pr, outfile);   % overwriting the file gives us advantage that we can loose the ;latitude and longitude information of Tmax,
% Tmin file and only work with time varible in 3rd dimension and then rewrite this over same nc file
```
3. Rename the internal variable (`pr`) of this copy with the name `adults`. It means, we have a new .nc file that contains
`longitude`, `latitude` and the variable name `adults`. With this we can just overwrite the variable adult with the calculations we intend to do.
```matlab
ncid = netcdf.open(outfile,'NC_WRITE');
netcdf.reDef(ncid)
netcdf.renameVar(ncid,3,'adults');
netcdf.endDef(ncid);
netcdf.close(ncid);
```

4. Defines a delta time step that will be important for solving the system of differential equations.
```matlab
step_t = 10;
```

5. Load variables from each dataset involved (`tmean`, `pr`, `dens`)
   
    a. Load temperature variables and applies some preprocessing steps with `load_temp2(tmean, step_t)` function
    ```matlab
    [Temp, Tmean] = load_temp2(tmean, step_t);       % Without DTR if Tmean is only available, no Tmax or Tmin
    ```

    b. Load population density variables and applies some preprocessing steps with `load_hdp(dens)`
    ```matlab
    DENS = load_hpd(dens);
    ```

    c. Load Precipitation variable with `load_rainfall(pr)` function
    ```matlab
    PR = load_rainfall(pr);    
    ```

    d. Load Latitude variable from the `tmean` dataset with the function `load_latitude(tmean)`
    ```matlab
    LAT = load_latitude(tmean);
    ```

6. Calculate **Juvenile Carrying Capacity**: $K_{L}(W,P)$

    - $W$: Rainfall accumulation, this the information we get from `PR`

    - $P$: Human density, this is the information we get from `DENS`

    ```matlab
    CC = capacity(PR, DENS);
    ```

      - file associated: `capacity.m`

      - Equation in Supplement Information: 14

7. Calculate **Hatching fraction depending in human density and rainfall**: $Q(W,P)$
    $$
    Q(W,P) = (1 - E_{rat}) \left( \frac{(1+E_{0})e^{\left(-E_{var}(W(t)-E_{opt})\right)^2}}{e^{\left( -E_{var} (W(t)-E_{opt})^2 \right)}} \right) + E_{rat} \left( \frac{E_{dens}}{E_{dens} + e^{-E_{fac}P}} \right)
    $$

    Where:

    - $W$: Precipitation, we get this information from `PR`

    - $P$: Human density, we get this information from `DENS`
    - $E_{opt}$ = 8;
    - $E_{var}$ = 0.05;
    - $E_{0}$ = 1.5;
    - $E_{rat}$= 0.2;
    - $E_{dens}$ = 0.01;
    - $E_{fac}$ = 0.01;
    
    ```matlab
    egg_active = water_hatch(PR, DENS);
    ```
    - file associated: `water_hatch.m`
    - Equation in Supplementary Information: 13

8. Create a **Vector Initial population**: $V_{0}$
    ```matlab
    previous = 'no_previous';
    v0 = load_initial(previous, size(Temp));
    ```
    - File associated: `load_initial.m`
    - TODO: Ask about the name and meaning of this variable.

9. Calculate the parameters for each time step and run ODEs
    ```matlab
    v = call_func(v0, Temp, Tmean, LAT, CC, egg_active, step_t);
    ```

    Input variables:

    - initial vector: `v0`
    - temperature: `Temp`
    - mean temperature: `Tmean`
    - latitude: `LAT`
    - juvenile carrying capacity: `CC`
    - hatching fraction depending in human density and rainfall: `egg_active`
    - time step: `step_time`
    - File associated: `call_func.m`

1.   Once all the variables are calculated, write to the output file.
```matlab
ncwrite(outfile,'adults', permute(v(:,:,5,:),[1,2,4,3]));
```

### Name: `call_func.m`

Description: This file the main file that contains the ODEs to calculate the 6 differential equations
proposed in the paper.

Input Variables:

- `v`
- `Temp` 
- `Tmean`
- `LAT`
- `CC`
- `egg_activate`
- `step_t`
  
Process:
  
  1. Calculate the **Diapause Lay**

     - File associated: `mosquito_dia_lay.m`
     - TODO: ask about the equation.

  2. Calculate the **Diapause Hatch**
     - File associated: `msoquito_dia_hatch.m`
     - TODO: ask about the equation

  3. Calculate **Diapausing egg mortality rate** : $m_{Ed}(T)$
    $$
    m_{Ed}(T) = -\ln\left(  0.955 \left(-0.5 \left(\frac{T-18.8}{21.53}\right)^6 \right) \right)
    $$

    - Input Variables: 
        - `Temp`
        - `step_t`
     - File associated: `mosq_surv_ed.m`
     - Comments: hardcoded values do not correspond to the values in the paper.

  4. Create a n-dimensional matrix `v_out` full of zeros.

  5. Create a for loop that iterates over the time:
   
    5.1. Creates a temperatures vector
		
    ```octave
    T = Temp(:,:,t);
    ```

    5.2. Calculates the Mosquito Birth

    $$
    mos
    $$

    - Inputs:
        - `T`
    - File associated: `mosq_birth.m`
  
    - Comments:
        - TODO: This equation is not present in the supplementary material.

    ```octave
    T = Temp(:,:,t);
    ```

    5.3. Calculates the Mosquito development j

    - Inputs:
        - `T`
    - File associated: `mosq_dev_j.m`
    - Comments:
        - TODO: This equation is not present in the suplement material.

    ```octave
    dev_j = mosq_dev_j(T);
    ```

    5.4. Calculates the Mosquito development i

    - Inputs:
        - `T`
    - File associated: `mosq_dev_i.m`
    - Comments:
        - TODO: This equation is not present in the suplement material.

    ```octave
    dev_j = mosq_dev_i(T);
    ```

    5.4. Other calculations I do not understand
    ```octave
    dev_e = 1./7.1;      %original function of the model 
    dia_lay = diapause_lay(:,:,ceil(t/step_t));
    dia_hatch = diapause_hatch(:,:,ceil(t/step_t));
    ed_surv = ed_survival(:,:,t);
    water_hatch = egg_activate(:,:,ceil(t/step_t));
    mort_e = mosq_mort_e(T);
    mort_j = mosq_mort_j(T);
    
    T = Tmean(:,:,ceil(t/step_t));
    mort_a = mosq_mort_a(T);
    ```

    5.5. ODE
    ```octave
    vars = {ceil(t/step_t), step_t, Temp, CC, birth, dia_lay, dia_hatch, mort_e, mort_j, mort_a, ed_surv, dev_j, dev_i, dev_e, water_hatch};
    v = RK4(@eqsys, @eqsys_log, v, vars, step_t); % Runge-Kutta 4 method
    % v = FE(@eqsys, @eqsys_log, v, vars, step_t); % Forward Euler method
    ```

    5.6. Additional calculations
    ```octave
    if mod(t/step_t,365) == 200
        v(:,:,2) = 0;
    end
    
    if mod(t,step_t) == 0
        if mod(ceil(t/step_t),30) == 0
            disp(['MOY: ', num2str(t/step_t/30)]);
        end
        for j = 1:5
            %v_out(:,:,j,t/step_t) = v(:,:,j);
            v_out(:,:,j,t/step_t) = max(v(:,:,j), 0);  % if any abundance is negative it will make it zero
        end
    end

    ```
---

## Table 1. Mapping Equations from Octave/Matlab code to Python code

!!! note

    Numbers in brackets refer to the corresponding row numbers in [Table S11 from the Supplementary Material](https://static-content.springer.com/esm/art%3A10.1038%2Fs43247-025-02199-z/MediaObjects/43247_2025_2199_MOESM2_ESM.pdf).

| Octave Function  | Python Function     | Equation Number in Paper |
| ---------------- | ------------------- | ------------------------ |
| `mosq_dev_j.m`   | `mosq_dev_j`        | [5]                      |
| `mosq_dev_i.m`   | `mosq_dev_i`        | [6]                      |
| `mosq_dev_e.m`   | `mosq_dev_e`        | `[NR 1]`                 |
| `capacity.m`     | `carrying_capacity` | [14]                     |
| `mosq_mort_e.m`  | `mosq_mort_e`       | [9]                      |
| `mosq_mort_j.m`  | `mosq_mort_j`       | [10]                     |
| `mosq_mort_a.m`  | `mosq_mort_a`       | [11]                     |
| `mosq_surv_ed.m` | `mosq_surv_ed`      | `[NR 2]`                 |



## Table 2. Climate sensitive parameter description and functions (inspired on Table S11 in Supplementary information)

!!! warning

    Put special attention to the Not Reported (`[NR <number>]`) equations and the Not Reported units (`[NA]`).


| Number   | Description                                          | Symbol         | Unit            |
| -------- | ---------------------------------------------------- | -------------- | --------------- |
| [5]      | Juvenile development rate                            |                | $\frac{1}{day}$ |
| [6]      | Emerging adult development                           |                | $\frac{1}{day}$ |
| `[NR 1]` | (tentative) Emerging adult development Briere model. |                | `[NR 1]`        |
| [14]     | Juvenile carrying capacity                           |                | `NA`            |
| [9]      | Egg mortality rate                                   | $m_{E}(T)$     | $\frac{1}{day}$ |
| [10]     | Juvenile mortality rate                              | $m_{J}(T)$     | $\frac{1}{day}$ |
| [11]     | Adult mortality rate                                 | $m_{A}(Tmean)$ | $\frac{1}{day}$ |
| `[NR 2]` | (tentative) Diapausing egg mortality rate            | $m_{Ed}(T)$    | $\frac{1}{day}$ |





## List of equations
- [5] Juvenile development rate: $\delta_{J}(T)$
$$
\delta_{J}(T) = \frac{1}{0.08T^{2} - 4.89T + 83.85}
$$

    | Parameter | Description      |
    | --------- | ---------------- |
    | $T$       | Temperature (°C) |

- [6] Emerging adult development rate: 
$$
\delta_{Aem}(T) = \frac{1}{0.069T^{2} - 3.574T + 50.1}
$$

    | Parameter | Description      |
    | --------- | ---------------- |
    | $T$       | Temperature (°C) |

- `[NR 1]` (tentative) Emerging adult development Briere model
$$
    BM = q \cdot T \cdot (T - T_0) \cdot \sqrt{T_m - T}
$$

    | Parameter | Description                                |
    | --------- | ------------------------------------------ |
    | $q$       | Empirical coefficient for development rate |
    | $T$       | Temperature (°C)                           |
    | $T_0$     | Minimum threshold temperature (°C)         |
    | $T_m$     | Maximum threshold temperature (°C)         |

- [14] Juvenile carrying capacity: 
$$
K_{L}(W,P) = \lambda\,\frac{0.1}{1 - 0.9^{t}}\sum_{x=1}^{t} 0.9^{(t-x)}\left(\alpha_{\text{rain}}W(x) + \alpha_{\text{dens}}P(x)\right)
$$

    | Parameter              | Description                        |
    | ---------------------- | ---------------------------------- |
    | $\lambda$              | Scaling coefficient                |
    | $t$                    | ---                                |
    | $x$                    | ---                                |
    | $\alpha_{\text{rain}}$ | Weight for rainfall contribution   |
    | $W(x)$                 | Rainfall at time step $x$          |
    | $\alpha_{\text{dens}}$ | Weight for population contribution |
    | $P(x)$                 | Population at time step $x$        |

- [9] Egg mortality rate:

!!! warning

    Equation in paper do not show the exponential function $exp()$. However, the exponential function is used in the Octave code.

$$
m_{E}(T) = -\ln\left( 0.955\, \exp\left[ -0.5 \left( \frac{T - 18.8}{21.53} \right)^{6} \right] \right)
$$

| Parameter | Description      |
| --------- | ---------------- |
| $T$       | Temperature (°C) |

- [10] Juvenile mortality rate:

!!! warning

    Equation in paper do not show the exponential function $exp()$. However, the exponential function is used in the Octave code.

$$
m_{J}(T) = -\ln\left[\,0.977\, \exp\left(-0.5 \left(\frac{T - 21.8}{16.6}\right)^{6}\right)\,\right]
$$

| Parameter | Description      |
| --------- | ---------------- |
| $T$       | Temperature (°C) |

- [11] Adult mortality rate:

!!! warning

    The equation reported on paper do not show the exponential $exp()$ found in the Octave/Matlab code. Additionally the equation in Octave should look like this.

    $$
    m_{A}(T) = 
    \begin{cases}
    -\ln\left[\,0.677\, \exp\left(-0.5 \left(\frac{T - 20.9}{13.2}\right)^{6}\right)\, T^{0.1}\,\right], & \text{if } T > 0 \\[1.5ex]
    -\ln\left[\,0.677\, \exp\left(-0.5 \left(\frac{T - 20.9}{13.2}\right)^{6}\right)\,\right], & \text{if } T \leq 0
    \end{cases}
    $$

$$
m_{A}(T_{\text{mean}}) = -\ln\left[\,0.677\, \exp\left(-0.5 \left(\frac{T_{\text{mean}} - 20.9}{13.2}\right)^{6}\right)\, (T_{\text{mean}})^{0.1}\,\right]
$$


| Parameter         | Description      |
| ----------------- | ---------------- |
| $T_{\text{mean}}$ | Temperature (°C) |

- `[NR 2]` (tentative) Diapausing egg mortality rate:
!!! warning
    The constants in the code are different than constants reported on the paper.

$$
m_{Ed}(T) = m_{E}(T) = -\ln\left( 0.955\, \exp\left[ -0.5 \left( \frac{T - 18.8}{21.53} \right)^{6} \right] \right)
$$

| Parameter | Description      |
| --------- | ---------------- |
| $T$       | Temperature (°C) |

