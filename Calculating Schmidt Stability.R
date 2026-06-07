# Schmidt Stability 
#This script has been designed for wide-format datasets

# ---- Load required Packages ----
pkgs <- c("readxl", "dplyr", "lubridate", "rLakeAnalyzer", "ggplot2")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) install.packages(to_install)
invisible(lapply(pkgs, library, character.only = TRUE))

# ---- Settings ----
lake_name <- "Rotorua"
bathy_max_depth <- 20.5 #Our water temp measurements only goes down to 20.5
min_points <- 5  #a profile must have at least 5 valid temperature points
min_span_m <- 2 #valid temperature points must cover at least 2 m of depth.

out_csv  <- "schmidt_stability_simple_gap_filled.csv"
out_plot <- "schmidt_stability_simple_gap_filled.png"

# ---- Helper: force NZ timezone without shifting clock time ----
ensure_nz_time_no_shift <- function(x, tz = "Pacific/Auckland") {
  if (inherits(x, "POSIXt")) {
    return(lubridate::force_tz(x, tzone = tz))
  } else {
    x_chr <- as.character(x)
    dt <- suppressWarnings(lubridate::ymd_hms(x_chr, tz = tz))
    if (all(is.na(dt))) dt <- suppressWarnings(lubridate::ymd_hm(x_chr, tz = tz))
    if (all(is.na(dt))) dt <- suppressWarnings(lubridate::dmy_hms(x_chr, tz = tz))
    if (all(is.na(dt))) dt <- suppressWarnings(lubridate::dmy_hm(x_chr, tz = tz))
    return(dt)
  }
}

# ---- Helper: Schmidt stability for one profile ----
#In our wide dataset, each row is one profile (water_temp_0.5, water_temp_1.0,...)
calc_S <- function(depth, temp, bthA, bthD, min_points = 5, min_span_m = 2) {
  ok <- !is.na(depth) & !is.na(temp)
  depth <- depth[ok]
  temp  <- temp[ok]
  
  if (length(depth) < min_points) return(NA_real_)
  
  o <- order(depth) #This makes sure the depths are ordered from shallow to deep
  depth <- depth[o]
  temp  <- temp[o]
  
  if ((max(depth) - min(depth)) < min_span_m) return(NA_real_)
  
  keep <- depth <= max(bthD)
  depth <- depth[keep]
  temp  <- temp[keep]
  
  if (length(depth) < min_points) return(NA_real_)
  if ((max(depth) - min(depth)) < min_span_m) return(NA_real_)
  #the most important part of the script: schmidt stability calculation
  s <- as.numeric(rLakeAnalyzer::schmidt.stability(
    wtr = temp,
    depths = depth,
    bthA = bthA,
    bthD = bthD,
    sal = 0 #Because of being a fresh water
  ))
  
  if (!is.na(s) && s < 0) s <- 0
  return(s) #defining physical restrictions: avoid negative stability
}

count_valid_points <- function(temp, depth_limit = Inf, depths) {
  sum(!is.na(temp) & depths <= depth_limit)
}

get_zmin <- function(temp, depth_limit = Inf, depths) {
  d <- depths[!is.na(temp) & depths <= depth_limit]
  if (length(d) == 0) return(NA_real_)
  min(d)
}

get_zmax <- function(temp, depth_limit = Inf, depths) {
  d <- depths[!is.na(temp) & depths <= depth_limit]
  if (length(d) == 0) return(NA_real_)
  max(d)
}

# 1) Read bathymetry Excel

message("Choose bathymetry Excel file...")
bathy <- readxl::read_excel(file.choose())

area_col <- dplyr::case_when(
  "MODEL SURFACE AREA (m2)" %in% names(bathy) ~ "MODEL SURFACE AREA (m2)",
  "PLANAR SURFACE AREA(m2)" %in% names(bathy) ~ "PLANAR SURFACE AREA(m2)",
  TRUE ~ NA_character_
)

if (is.na(area_col)) {
  stop("Bathymetry must contain either 'MODEL SURFACE AREA (m2)' or 'PLANAR SURFACE AREA(m2)'.")
}

bth <- bathy %>%
  { if ("LAKE" %in% names(.)) dplyr::filter(., LAKE == lake_name) else . } %>%
  transmute(
    bthD = abs(`DEPTH (m)`),
    bthA = .data[[area_col]]
  ) %>%
  filter(!is.na(bthD), !is.na(bthA)) %>%
  arrange(bthD)

if (is.finite(bathy_max_depth)) {
  bth <- bth %>% filter(bthD <= bathy_max_depth)
}

if (nrow(bth) == 0) stop("No bathymetry data left after depth filtering.")
if (!any(bth$bthD == 0)) stop("Bathymetry must include depth = 0.")
if (any(diff(bth$bthD) <= 0)) stop("Bathymetry depths must increase strictly.")
if (any(bth$bthA <= 0)) stop("Bathymetry areas must be positive.")

bthD <- bth$bthD
bthA <- bth$bthA

message("Bathymetry depth range used: 0 to ", max(bthD), " m")

# 2) Read temperature Excel

message("Choose temperature Excel file...")
prof <- readxl::read_excel(file.choose())

if (!"datetime" %in% names(prof)) {
  stop("Temperature file must contain a 'datetime' column.")
}

temp_cols <- c(
  "Water_Temp_0.5", "Water_Temp_1.0", "Water_Temp_2.5", "Water_Temp_4.5",
  "Water_Temp_6.5", "Water_Temp_8.5", "Water_Temp_10.5", "Water_Temp_12.5",
  "Water_Temp_14.5", "Water_Temp_16.5", "Water_Temp_18.5", "Water_Temp_20.5"
)

missing_temp_cols <- setdiff(temp_cols, names(prof))
if (length(missing_temp_cols) > 0) {
  stop("Temperature file missing columns: ", paste(missing_temp_cols, collapse = ", "))
}

depths <- c(0.5, 1.0, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5)

prof <- prof %>%
  mutate(
    datetime = ensure_nz_time_no_shift(datetime, tz = "Pacific/Auckland")
  ) #This converts the datetime column into a proper R datetime object.

if (all(is.na(prof$datetime))) stop("datetime could not be parsed.")

# 3) Compute Schmidt stability per time step

depth_limit <- max(bthD)

stability_by_time <- prof %>%
  rowwise() %>% #calculate row by row.
  mutate(
    n = count_valid_points(
      temp = c_across(all_of(temp_cols)),
      depth_limit = depth_limit,
      depths = depths
    ),
    zmin = get_zmin(
      temp = c_across(all_of(temp_cols)),
      depth_limit = depth_limit,
      depths = depths
    ),
    zmax = get_zmax(
      temp = c_across(all_of(temp_cols)),
      depth_limit = depth_limit,
      depths = depths
    ),
    S = calc_S(
      depth = depths,
      temp  = c_across(all_of(temp_cols)),
      bthA  = bthA,
      bthD  = bthD,
      min_points = min_points,
      min_span_m = min_span_m
    )
  ) %>%
  ungroup() %>%
  mutate(
    time = datetime
  )

# 4) Plot 
# This makes a time-series plot of Schmidt stability

p <- ggplot(stability_by_time %>% filter(!is.na(S)),
            aes(x = time, y = S)) +
  geom_line() +
  geom_point(size = 0.4) +
  labs(
    title = "Schmidt Stability",
    x = "Time",
    y = "Schmidt Stability (J/m²)"
  )

print(p)
ggsave(out_plot, plot = p, width = 10, height = 5, dpi = 150)

# 5) Save CSV

stability_out <- stability_by_time %>%
  select(time, n, zmin, zmax, S) %>%
  mutate(
    time = format(time, "%d/%m/%Y %H:%M:%S")
  )

write.csv(stability_out, out_csv, row.names = FALSE)

message("Done!")
message("Saved CSV:  ", file.path(getwd(), out_csv))
message("Saved plot: ", file.path(getwd(), out_plot))