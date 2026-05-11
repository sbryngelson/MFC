#define _GNU_SOURCE

#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MFC_TRACE_STACK_MAX 4096
#define MFC_TRACE_NAME_MAX 512
#define MFC_TRACE_SYMBOL_CACHE_MAX 512
#define MFC_TRACE_SETUP_MAX 8192
#define MFC_TRACE_CONTEXT_MAX 16384
#define MFC_TRACE_COLOR_RESET "\033[0m"
#define MFC_TRACE_COLOR_BOLD "\033[1m"
#define MFC_TRACE_COLOR_DIM "\033[2m"
#define MFC_TRACE_COLOR_TRACE "\033[32m"
#define MFC_TRACE_COLOR_CONTEXT "\033[36m"
#define MFC_TRACE_COLOR_MPI "\033[35m"

typedef struct {
    void *addr;
    char name[MFC_TRACE_NAME_MAX];
} mfc_trace_symbol_cache_entry;

typedef struct {
    void *addr;
    void *call_site;
    uint64_t self_id;
    uint64_t parent_id;
    int depth;
    int emitted;
} mfc_trace_setup_entry;

typedef struct {
    void *addr;
    void *call_site;
} mfc_trace_context_entry;

static __thread void *trace_stack[MFC_TRACE_STACK_MAX];
static __thread void *trace_stack_call_sites[MFC_TRACE_STACK_MAX];
static __thread uint64_t trace_stack_ids[MFC_TRACE_STACK_MAX];
static __thread uint64_t trace_last_dump_ids[MFC_TRACE_STACK_MAX];
static __thread mfc_trace_symbol_cache_entry trace_symbol_cache[MFC_TRACE_SYMBOL_CACHE_MAX];
static __thread mfc_trace_setup_entry trace_setup[MFC_TRACE_SETUP_MAX];
static __thread mfc_trace_context_entry trace_context[MFC_TRACE_CONTEXT_MAX];
static __thread mfc_trace_context_entry trace_mpi_context[MFC_TRACE_CONTEXT_MAX];
static __thread int trace_stack_depth = 0;
static __thread int trace_last_dump_depth = -1;
static __thread int trace_point_depth = 0;
static __thread int trace_in_hook = 0;
static __thread int trace_symbol_cache_count = 0;
static __thread int trace_setup_count = 0;
static __thread int trace_context_count = 0;
static __thread int trace_mpi_context_count = 0;
static __thread uint64_t trace_next_stack_id = 1;
static pthread_once_t trace_initialize_once = PTHREAD_ONCE_INIT;
static int trace_enabled = 0;
static int trace_point_enabled = 0;
static int trace_include_setup = 0;
static int trace_context_enabled = 0;
static int trace_mpi_enabled = 0;
static int trace_tree_format = 0;
static int trace_color_enabled = 0;
static int trace_stdout_enabled = 0;
static int trace_file_fd = -1;

static void mfc_trace_initialize_once(void) __attribute__((no_instrument_function));
static void mfc_trace_initialize(void) __attribute__((no_instrument_function));
static int mfc_trace_process_rank(void) __attribute__((no_instrument_function));
static void mfc_trace_executable_name(char *buffer, size_t buffer_size) __attribute__((no_instrument_function));
static void mfc_trace_format_function_name(const char *name, char *buffer, size_t buffer_size) __attribute__((no_instrument_function));
static void mfc_trace_pretty_file(const char *file, char *buffer, size_t buffer_size) __attribute__((no_instrument_function));
static void mfc_trace_shell_quote(const char *text, char *buffer, size_t buffer_size) __attribute__((no_instrument_function));
static int mfc_trace_symbol_from_addr2line(void *addr, const char *image, char *buffer, size_t buffer_size) __attribute__((no_instrument_function));
static int mfc_trace_call_site_from_addr2line(void *addr, char *buffer, size_t buffer_size) __attribute__((no_instrument_function));
static void mfc_trace_symbol(void *addr, char *buffer, size_t buffer_size) __attribute__((no_instrument_function));
static int mfc_trace_skip_symbol(const char *name) __attribute__((no_instrument_function));
static int mfc_trace_is_mpi_symbol(const char *name) __attribute__((no_instrument_function));
static const char *mfc_trace_prefix_color(const char *prefix) __attribute__((no_instrument_function));
static int mfc_trace_format_line(char *line, size_t line_size, int color, const char *prefix, const char *name, const char *caller, int depth)
    __attribute__((no_instrument_function));
static void mfc_trace_write_at_depth(const char *name, void *call_site, int depth) __attribute__((no_instrument_function));
static void mfc_trace_write_line(const char *prefix, const char *name, void *call_site, int depth) __attribute__((no_instrument_function));
static void mfc_trace_remember_setup_call(void *addr, void *call_site) __attribute__((no_instrument_function));
static int mfc_trace_remember_context_call(void *addr, void *call_site) __attribute__((no_instrument_function));
static void mfc_trace_write_context_call(void *addr, void *call_site) __attribute__((no_instrument_function));
static int mfc_trace_remember_mpi_call(void *addr, void *call_site) __attribute__((no_instrument_function));
static void mfc_trace_write_mpi_call(void *addr, void *call_site) __attribute__((no_instrument_function));
static int mfc_trace_stack_index_for_id(uint64_t id) __attribute__((no_instrument_function));
static void mfc_trace_flush_setup_for_parent(uint64_t parent_id) __attribute__((no_instrument_function));
static int mfc_trace_common_dumped_prefix(void) __attribute__((no_instrument_function));
static void mfc_trace_remember_dumped_stack(void) __attribute__((no_instrument_function));
static void mfc_trace_dump_stack(void) __attribute__((no_instrument_function));

static void mfc_trace_initialize_once(void) {
    const char *trace_env;
    const char *point_env;
    const char *setup_env;
    const char *context_env;
    const char *mpi_env;
    const char *format_env;
    const char *color_env;
    const char *term_env;
    const char *file_env;
    const char *stdout_env;

    trace_env = getenv("MFC_TRACE");
    trace_enabled = trace_env != NULL && trace_env[0] != '\0' && strcmp(trace_env, "0") != 0;
    if (mfc_trace_process_rank() != 0) trace_enabled = 0;

    point_env = getenv("MFC_TRACE_POINT");
    trace_point_enabled = point_env != NULL && point_env[0] != '\0';

    setup_env = getenv("MFC_TRACE_INCLUDE_SETUP");
    trace_include_setup = setup_env != NULL && setup_env[0] != '\0' && strcmp(setup_env, "0") != 0;

    context_env = getenv("MFC_TRACE_CONTEXT");
    trace_context_enabled = context_env != NULL && context_env[0] != '\0' && strcmp(context_env, "0") != 0;

    mpi_env = getenv("MFC_TRACE_MPI");
    trace_mpi_enabled = mpi_env != NULL && mpi_env[0] != '\0' && strcmp(mpi_env, "0") != 0;

    format_env = getenv("MFC_TRACE_FORMAT");
    trace_tree_format = format_env != NULL && (strcmp(format_env, "tree") == 0 || strcmp(format_env, "structured") == 0);

    color_env = getenv("MFC_TRACE_COLOR");
    term_env = getenv("TERM");
    stdout_env = getenv("MFC_TRACE_STDOUT");
    if (stdout_env != NULL && stdout_env[0] != '\0') {
        trace_stdout_enabled = strcmp(stdout_env, "0") != 0 && strcmp(stdout_env, "false") != 0 && strcmp(stdout_env, "never") != 0;
    } else {
        trace_stdout_enabled = isatty(STDOUT_FILENO);
    }
    if (color_env != NULL && (strcmp(color_env, "always") == 0 || strcmp(color_env, "1") == 0 || strcmp(color_env, "true") == 0)) {
        trace_color_enabled = trace_stdout_enabled;
    } else if (color_env != NULL && (strcmp(color_env, "never") == 0 || strcmp(color_env, "0") == 0 || strcmp(color_env, "false") == 0)) {
        trace_color_enabled = 0;
    } else {
        trace_color_enabled = trace_stdout_enabled && getenv("NO_COLOR") == NULL && term_env != NULL && strcmp(term_env, "dumb") != 0;
    }

    file_env = getenv("MFC_TRACE_FILE");
    if (trace_enabled && file_env != NULL && file_env[0] != '\0') {
        trace_file_fd = open(file_env, O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (trace_file_fd < 0) {
            static const char message[] = "MFC trace: failed to open MFC_TRACE_FILE\n";
            (void)write(STDERR_FILENO, message, sizeof(message) - 1);
        }
    }

    if (trace_enabled) {
        char executable[MFC_TRACE_NAME_MAX];
        mfc_trace_executable_name(executable, sizeof(executable));
        mfc_trace_write_line("TRACE_RUN", executable, NULL, 0);
    }
}

static void mfc_trace_initialize(void) {
    (void)pthread_once(&trace_initialize_once, mfc_trace_initialize_once);
}

static int mfc_trace_process_rank(void) {
    const char *rank_env_names[] = {
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "MV2_COMM_WORLD_RANK",
        "SLURM_PROCID",
        NULL,
    };
    const char *value;
    char *end;
    long rank;
    int i;

    for (i = 0; rank_env_names[i] != NULL; ++i) {
        value = getenv(rank_env_names[i]);
        if (value == NULL || value[0] == '\0') continue;

        rank = strtol(value, &end, 10);
        if (end != value) return (int)rank;
    }

    return 0;
}

static void mfc_trace_executable_name(char *buffer, size_t buffer_size) {
    char path[1024];
    const char *name;
    ssize_t len;

    if (buffer_size == 0) return;

    len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len > 0) {
        path[len] = '\0';
        name = strrchr(path, '/');
        snprintf(buffer, buffer_size, "%s", name != NULL ? name + 1 : path);
        return;
    }

    snprintf(buffer, buffer_size, "program");
}

static void mfc_trace_format_function_name(const char *name, char *buffer, size_t buffer_size) {
    const char *mod_marker;
    const char *mp_marker;
    const char *mod_start;
    const char *proc_start;
    const char *suffix;
    size_t mod_len;
    size_t proc_len;

    if (name == NULL || name[0] == '\0') {
        snprintf(buffer, buffer_size, "??");
        return;
    }

    mod_marker = strstr(name, "_MOD_");
    if (strncmp(name, "__", 2) == 0 && mod_marker != NULL) {
        mod_start = name + 2;
        proc_start = mod_marker + 5;
        mod_len = (size_t)(mod_marker - mod_start);
        proc_len = strlen(proc_start);
        snprintf(buffer, buffer_size, "%.*s:%.*s", (int)mod_len, mod_start, (int)proc_len, proc_start);
        return;
    }

    mp_marker = strstr(name, "_mp_");
    if (mp_marker != NULL) {
        mod_start = name;
        proc_start = mp_marker + 4;
        mod_len = (size_t)(mp_marker - mod_start);
        proc_len = strlen(proc_start);
        if (proc_len > 0 && proc_start[proc_len - 1] == '_') proc_len -= 1;
        snprintf(buffer, buffer_size, "%.*s_%.*s", (int)mod_len, mod_start, (int)proc_len, proc_start);
        return;
    }

    suffix = name + strlen(name);
    if (suffix > name && suffix[-1] == '_') {
        snprintf(buffer, buffer_size, "%.*s", (int)(suffix - name - 1), name);
        return;
    }

    snprintf(buffer, buffer_size, "%s", name);
}

static void mfc_trace_pretty_file(const char *file, char *buffer, size_t buffer_size) {
    const char *src;
    const char *build;

    if (file == NULL || file[0] == '\0' || strcmp(file, "??:0") == 0 || strcmp(file, "??:?") == 0) {
        buffer[0] = '\0';
        return;
    }

    src = strstr(file, "/src/");
    if (src != NULL) {
        snprintf(buffer, buffer_size, "%s", src + 1);
        return;
    }

    build = strstr(file, "/build/");
    if (build != NULL) {
        snprintf(buffer, buffer_size, "%s", build + 1);
        return;
    }

    snprintf(buffer, buffer_size, "%s", file);
}

static void mfc_trace_shell_quote(const char *text, char *buffer, size_t buffer_size) {
    size_t used = 0;
    size_t i;

    if (buffer_size == 0) return;

    buffer[used++] = '\'';
    for (i = 0; text != NULL && text[i] != '\0' && used + 5 < buffer_size; ++i) {
        if (text[i] == '\'') {
            buffer[used++] = '\'';
            buffer[used++] = '\\';
            buffer[used++] = '\'';
            buffer[used++] = '\'';
        } else {
            buffer[used++] = text[i];
        }
    }

    if (used + 1 < buffer_size) buffer[used++] = '\'';
    buffer[used < buffer_size ? used : buffer_size - 1] = '\0';
}

static int mfc_trace_symbol_from_addr2line(void *addr, const char *image, char *buffer, size_t buffer_size) {
    char quoted_image[1024];
    char command[1400];
    char raw_name[MFC_TRACE_NAME_MAX];
    char raw_file[MFC_TRACE_NAME_MAX];
    char pretty_name[MFC_TRACE_NAME_MAX];
    char pretty_file[MFC_TRACE_NAME_MAX];
    FILE *pipe;
    size_t len;

    if (image == NULL || image[0] == '\0') return 0;

    mfc_trace_shell_quote(image, quoted_image, sizeof(quoted_image));
    snprintf(command, sizeof(command), "addr2line -f -C -e %s %p 2>/dev/null", quoted_image, addr);

    pipe = popen(command, "r");
    if (pipe == NULL) return 0;

    if (fgets(raw_name, sizeof(raw_name), pipe) == NULL || fgets(raw_file, sizeof(raw_file), pipe) == NULL) {
        (void)pclose(pipe);
        return 0;
    }

    (void)pclose(pipe);

    len = strlen(raw_name);
    while (len > 0 && (raw_name[len - 1] == '\n' || raw_name[len - 1] == '\r')) raw_name[--len] = '\0';

    len = strlen(raw_file);
    while (len > 0 && (raw_file[len - 1] == '\n' || raw_file[len - 1] == '\r')) raw_file[--len] = '\0';

    if (raw_name[0] == '\0' || strcmp(raw_name, "??") == 0) return 0;

    mfc_trace_format_function_name(raw_name, pretty_name, sizeof(pretty_name));
    mfc_trace_pretty_file(raw_file, pretty_file, sizeof(pretty_file));

    if (pretty_file[0] != '\0') {
        snprintf(buffer, buffer_size, "%s [%s]", pretty_name, pretty_file);
    } else {
        snprintf(buffer, buffer_size, "%s", pretty_name);
    }

    return 1;
}

static int mfc_trace_call_site_from_addr2line(void *addr, char *buffer, size_t buffer_size) {
    Dl_info info;
    char quoted_image[1024];
    char command[1400];
    char raw_name[MFC_TRACE_NAME_MAX];
    char raw_file[MFC_TRACE_NAME_MAX];
    char pretty_name[MFC_TRACE_NAME_MAX];
    char pretty_file[MFC_TRACE_NAME_MAX];
    FILE *pipe;
    size_t len;

    if (addr == NULL) return 0;
    if (dladdr(addr, &info) == 0 || info.dli_fname == NULL) return 0;

    mfc_trace_shell_quote(info.dli_fname, quoted_image, sizeof(quoted_image));
    snprintf(command, sizeof(command), "addr2line -f -C -e %s %p 2>/dev/null", quoted_image, addr);

    pipe = popen(command, "r");
    if (pipe == NULL) return 0;

    if (fgets(raw_name, sizeof(raw_name), pipe) == NULL || fgets(raw_file, sizeof(raw_file), pipe) == NULL) {
        (void)pclose(pipe);
        return 0;
    }

    (void)pclose(pipe);

    len = strlen(raw_name);
    while (len > 0 && (raw_name[len - 1] == '\n' || raw_name[len - 1] == '\r')) raw_name[--len] = '\0';

    len = strlen(raw_file);
    while (len > 0 && (raw_file[len - 1] == '\n' || raw_file[len - 1] == '\r')) raw_file[--len] = '\0';

    if (raw_file[0] == '\0' || strcmp(raw_file, "??:0") == 0 || strcmp(raw_file, "??:?") == 0) return 0;

    mfc_trace_format_function_name(raw_name, pretty_name, sizeof(pretty_name));
    mfc_trace_pretty_file(raw_file, pretty_file, sizeof(pretty_file));

    if (pretty_name[0] != '\0' && strcmp(pretty_name, "??") != 0) {
        snprintf(buffer, buffer_size, " <- called from %s [%s]", pretty_name, pretty_file);
    } else {
        snprintf(buffer, buffer_size, " <- called from [%s]", pretty_file);
    }

    return 1;
}

static void mfc_trace_symbol(void *addr, char *buffer, size_t buffer_size) {
    Dl_info info;
    const char *image = NULL;
    char resolved[MFC_TRACE_NAME_MAX];
    char dladdr_name[MFC_TRACE_NAME_MAX];
    int dladdr_ok;
    int i;

    for (i = 0; i < trace_symbol_cache_count; ++i) {
        if (trace_symbol_cache[i].addr == addr) {
            snprintf(buffer, buffer_size, "%s", trace_symbol_cache[i].name);
            return;
        }
    }

    dladdr_ok = dladdr(addr, &info);
    dladdr_name[0] = '\0';
    if (dladdr_ok != 0) {
        image = info.dli_fname;
        if (info.dli_sname != NULL) {
            mfc_trace_format_function_name(info.dli_sname, dladdr_name, sizeof(dladdr_name));
        }
    }

    if (mfc_trace_symbol_from_addr2line(addr, image, resolved, sizeof(resolved))) {
        snprintf(buffer, buffer_size, "%s", resolved);
    } else if (dladdr_name[0] != '\0') {
        snprintf(buffer, buffer_size, "%s", dladdr_name);
    } else {
        snprintf(buffer, buffer_size, "%p", addr);
    }

    if (trace_symbol_cache_count < MFC_TRACE_SYMBOL_CACHE_MAX) {
        trace_symbol_cache[trace_symbol_cache_count].addr = addr;
        snprintf(trace_symbol_cache[trace_symbol_cache_count].name, sizeof(trace_symbol_cache[trace_symbol_cache_count].name), "%s", buffer);
        trace_symbol_cache_count += 1;
    }
}

static int mfc_trace_skip_symbol(const char *name) {
    return strstr(name, "__cyg_profile_func_") != NULL ||
           strstr(name, "mfc_trace_") != NULL ||
           strstr(name, "m_trace:") != NULL ||
           strstr(name, "m_trace_s_") != NULL ||
           strstr(name, "m_nvtx_") != NULL ||
           strncmp(name, "..acc_", 6) == 0 ||
           strstr(name, "acc_cuda_funcreg_constructor") != NULL ||
           strstr(name, "acc_data_constructor") != NULL ||
           strcmp(name, "s_trace_point_begin") == 0 ||
           strcmp(name, "s_trace_point_end") == 0;
}

static int mfc_trace_is_mpi_symbol(const char *name) {
    return strstr(name, "m_mpi") != NULL ||
           strstr(name, "_mpi_") != NULL ||
           strstr(name, ":mpi_") != NULL ||
           strstr(name, "mpi_") == name;
}

static const char *mfc_trace_prefix_color(const char *prefix) {
    if (strcmp(prefix, "TRACE_CONTEXT") == 0) return MFC_TRACE_COLOR_CONTEXT;
    if (strcmp(prefix, "TRACE_MPI") == 0) return MFC_TRACE_COLOR_MPI;
    return MFC_TRACE_COLOR_TRACE;
}

static void mfc_trace_write_at_depth(const char *name, void *call_site, int depth) {
    mfc_trace_write_line("TRACE", name, call_site, depth);
}

static int mfc_trace_format_line(char *line, size_t line_size, int color, const char *prefix, const char *name, const char *caller, int depth) {
    int len = 0;
    int i;

    if (color) {
        const char *prefix_color = mfc_trace_prefix_color(prefix);

        len = snprintf(line, line_size, "%s%s%s ", prefix_color, prefix, MFC_TRACE_COLOR_RESET);
        if (trace_tree_format) {
            for (i = 0; i < depth && len + 2 < (int)line_size; ++i) {
                line[len++] = ' ';
                line[len++] = ' ';
            }
        }
        if (caller[0] != '\0') {
            len += snprintf(line + len, line_size - (size_t)len, "%s%s%s%s%s%s\n", MFC_TRACE_COLOR_BOLD, name,
                            MFC_TRACE_COLOR_RESET, MFC_TRACE_COLOR_DIM, caller, MFC_TRACE_COLOR_RESET);
        } else {
            len += snprintf(line + len, line_size - (size_t)len, "%s%s%s\n", MFC_TRACE_COLOR_BOLD, name, MFC_TRACE_COLOR_RESET);
        }
    } else if (!trace_tree_format) {
        len = snprintf(line, line_size, "%s %s%s\n", prefix, name, caller);
    } else {
        len = snprintf(line, line_size, "%s ", prefix);
        for (i = 0; i < depth && len + 2 < (int)line_size; ++i) {
            line[len++] = ' ';
            line[len++] = ' ';
        }
        if (len < (int)line_size) {
            len += snprintf(line + len, line_size - (size_t)len, "%s%s\n", name, caller);
        }
    }

    if (len > 0 && (size_t)len > line_size) len = (int)line_size;
    return len;
}

static void mfc_trace_write_line(const char *prefix, const char *name, void *call_site, int depth) {
    char line[MFC_TRACE_NAME_MAX * 2 + 256];
    char caller[MFC_TRACE_NAME_MAX];
    int len;

    if (mfc_trace_skip_symbol(name)) return;
    caller[0] = '\0';
    (void)mfc_trace_call_site_from_addr2line(call_site, caller, sizeof(caller));

    len = mfc_trace_format_line(line, sizeof(line), 0, prefix, name, caller, depth);
    if (len > 0) {
        if (trace_file_fd >= 0) (void)write(trace_file_fd, line, (size_t)len);
        if (trace_stdout_enabled) {
            len = mfc_trace_format_line(line, sizeof(line), trace_color_enabled, prefix, name, caller, depth);
            if (len > 0) (void)write(STDOUT_FILENO, line, (size_t)len);
        }
    }
}

static void mfc_trace_remember_setup_call(void *addr, void *call_site) {
    int parent_depth;
    uint64_t parent_id;
    int i;

    if (!trace_include_setup || !trace_point_enabled || trace_point_depth > 0) return;
    if (trace_stack_depth < 2 || trace_setup_count >= MFC_TRACE_SETUP_MAX) return;

    parent_depth = trace_stack_depth - 2;
    if (parent_depth < 2) return;
    parent_id = trace_stack_ids[parent_depth];

    for (i = 0; i < trace_setup_count; ++i) {
        if (trace_setup[i].parent_id == parent_id && trace_setup[i].addr == addr) return;
    }

    trace_setup[trace_setup_count].addr = addr;
    trace_setup[trace_setup_count].call_site = call_site;
    trace_setup[trace_setup_count].self_id = trace_stack_ids[trace_stack_depth - 1];
    trace_setup[trace_setup_count].parent_id = parent_id;
    trace_setup[trace_setup_count].depth = trace_stack_depth - 1;
    trace_setup[trace_setup_count].emitted = 0;
    trace_setup_count += 1;
}

static int mfc_trace_remember_context_call(void *addr, void *call_site) {
    int i;

    if (!trace_context_enabled || !trace_point_enabled || trace_point_depth > 0) return 0;
    if (trace_context_count >= MFC_TRACE_CONTEXT_MAX) return 0;

    for (i = 0; i < trace_context_count; ++i) {
        if (trace_context[i].addr == addr && trace_context[i].call_site == call_site) return 0;
    }

    trace_context[trace_context_count].addr = addr;
    trace_context[trace_context_count].call_site = call_site;
    trace_context_count += 1;
    return 1;
}

static void mfc_trace_write_context_call(void *addr, void *call_site) {
    char name[MFC_TRACE_NAME_MAX];

    if (!mfc_trace_remember_context_call(addr, call_site)) return;

    mfc_trace_symbol(addr, name, sizeof(name));
    mfc_trace_write_line("TRACE_CONTEXT", name, call_site, trace_stack_depth > 0 ? trace_stack_depth - 1 : 0);
}

static int mfc_trace_remember_mpi_call(void *addr, void *call_site) {
    int i;

    if (!trace_mpi_enabled || trace_mpi_context_count >= MFC_TRACE_CONTEXT_MAX) return 0;

    for (i = 0; i < trace_mpi_context_count; ++i) {
        if (trace_mpi_context[i].addr == addr && trace_mpi_context[i].call_site == call_site) return 0;
    }

    trace_mpi_context[trace_mpi_context_count].addr = addr;
    trace_mpi_context[trace_mpi_context_count].call_site = call_site;
    trace_mpi_context_count += 1;
    return 1;
}

static void mfc_trace_write_mpi_call(void *addr, void *call_site) {
    char name[MFC_TRACE_NAME_MAX];

    mfc_trace_symbol(addr, name, sizeof(name));
    if (!mfc_trace_is_mpi_symbol(name)) return;
    if (!mfc_trace_remember_mpi_call(addr, call_site)) return;

    mfc_trace_write_line("TRACE_MPI", name, call_site, trace_stack_depth > 0 ? trace_stack_depth - 1 : 0);
}

static int mfc_trace_stack_index_for_id(uint64_t id) {
    int i;

    for (i = 0; i < trace_stack_depth; ++i) {
        if (trace_stack_ids[i] == id) return i;
    }

    return -1;
}

static void mfc_trace_flush_setup_for_parent(uint64_t parent_id) {
    char name[MFC_TRACE_NAME_MAX];
    int i;

    if (!trace_include_setup) return;

    for (i = 0; i < trace_setup_count; ++i) {
        if (trace_setup[i].emitted) continue;
        if (trace_setup[i].parent_id != parent_id) continue;
        if (mfc_trace_stack_index_for_id(trace_setup[i].self_id) >= 0) continue;

        mfc_trace_symbol(trace_setup[i].addr, name, sizeof(name));
        mfc_trace_write_at_depth(name, trace_setup[i].call_site, trace_setup[i].depth);
        trace_setup[i].emitted = 1;
    }
}

static int mfc_trace_common_dumped_prefix(void) {
    int i;
    int prefix = 0;
    int depth = trace_last_dump_depth;

    if (depth < 0) return 0;
    if (depth > trace_stack_depth) depth = trace_stack_depth;

    for (i = 0; i < depth; ++i) {
        if (trace_last_dump_ids[i] != trace_stack_ids[i]) break;
        prefix += 1;
    }

    return prefix;
}

static void mfc_trace_remember_dumped_stack(void) {
    int i;

    trace_last_dump_depth = trace_stack_depth;
    for (i = 0; i < trace_stack_depth; ++i) {
        trace_last_dump_ids[i] = trace_stack_ids[i];
    }
}

static void mfc_trace_dump_stack(void) {
    char name[MFC_TRACE_NAME_MAX];
    int i;
    int start;

    start = mfc_trace_common_dumped_prefix();

    for (i = start; i < trace_stack_depth; ++i) {
        mfc_trace_symbol(trace_stack[i], name, sizeof(name));
        mfc_trace_write_at_depth(name, trace_stack_call_sites[i], i);
        mfc_trace_flush_setup_for_parent(trace_stack_ids[i]);
    }

    mfc_trace_remember_dumped_stack();
}

void mfc_trace_point_begin(void) {
    mfc_trace_initialize();
    if (!trace_enabled || !trace_point_enabled) return;

    if (trace_point_depth == 0) {
        trace_in_hook = 1;
        mfc_trace_dump_stack();
        trace_in_hook = 0;
    }

    trace_point_depth += 1;
}

void mfc_trace_point_end(void) {
    mfc_trace_initialize();
    if (!trace_enabled || !trace_point_enabled) return;
    if (trace_point_depth > 0) trace_point_depth -= 1;
}

void __cyg_profile_func_enter(void *this_fn, void *call_site) {
    char name[MFC_TRACE_NAME_MAX];
    (void)call_site;

    if (trace_in_hook) return;
    trace_in_hook = 1;

    if (trace_stack_depth < MFC_TRACE_STACK_MAX) {
        trace_stack_ids[trace_stack_depth] = trace_next_stack_id++;
        trace_stack_call_sites[trace_stack_depth] = call_site;
        trace_stack[trace_stack_depth++] = this_fn;
    }

    mfc_trace_initialize();
    mfc_trace_remember_setup_call(this_fn, call_site);
    if (trace_enabled) {
        mfc_trace_write_context_call(this_fn, call_site);
        mfc_trace_write_mpi_call(this_fn, call_site);
    }

    if (trace_enabled && (!trace_point_enabled || trace_point_depth > 0)) {
        mfc_trace_symbol(this_fn, name, sizeof(name));
        mfc_trace_write_at_depth(name, call_site, trace_stack_depth > 0 ? trace_stack_depth - 1 : 0);
    }

    trace_in_hook = 0;
}

void __cyg_profile_func_exit(void *this_fn, void *call_site) {
    (void)this_fn;
    (void)call_site;

    if (trace_in_hook) return;
    if (trace_stack_depth > 0) trace_stack_depth -= 1;
}
