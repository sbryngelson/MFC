!>
!! @file
!! @brief Contains module m_trace

!> @brief Simple runtime call tracing helpers.
module m_trace

    use iso_fortran_env, only: output_unit

    implicit none

    private
    public :: s_trace_enter, s_trace_call, s_trace_global_call, s_trace_point_begin, s_trace_point_end

    logical, private           :: trace_initialized = .false.
    logical, private           :: trace_enabled = .false.
    logical, private           :: trace_point_enabled = .false.
    logical, private           :: trace_point_middle = .false.
    integer, private           :: trace_j = 0
    integer, private           :: trace_k = 0
    integer, private           :: trace_l = 0
    integer, private           :: trace_point_depth = 0
    character(len=64), private :: trace_point_vars = ''

    interface
        subroutine c_trace_point_begin() bind(C, name="mfc_trace_point_begin")

        end subroutine c_trace_point_begin

        subroutine c_trace_point_end() bind(C, name="mfc_trace_point_end")

        end subroutine c_trace_point_end
    end interface

contains

    !> Initialize trace settings from environment variables.
    subroutine s_initialize_trace

        character(len=64) :: env_value
        integer           :: env_len
        integer           :: first_comma
        integer           :: second_comma
        integer           :: read_status

        if (trace_initialized) return

        call get_environment_variable('MFC_TRACE', env_value, length=env_len)
        trace_enabled = env_len > 0 .and. trim(env_value) /= '0'

        call get_environment_variable('MFC_TRACE_POINT', env_value, length=env_len)
        if (env_len > 0) then
            if (trim(env_value(:env_len)) == 'middle') then
                trace_point_enabled = .true.
                trace_point_middle = .true.
            else
                first_comma = index(env_value, ',')
                second_comma = 0
                if (first_comma > 0) second_comma = index(env_value(first_comma + 1:), ',')
                if (second_comma > 0) second_comma = second_comma + first_comma

                if (first_comma > 0 .and. second_comma > first_comma) then
                    read (env_value(:first_comma - 1), *, iostat=read_status) trace_j
                    trace_point_enabled = read_status == 0
                    read (env_value(first_comma + 1:second_comma - 1), *, iostat=read_status) trace_k
                    trace_point_enabled = trace_point_enabled .and. read_status == 0
                    read (env_value(second_comma + 1:env_len), *, iostat=read_status) trace_l
                    trace_point_enabled = trace_point_enabled .and. read_status == 0
                end if
            end if
        end if

        call get_environment_variable('MFC_TRACE_POINT_VARS', env_value, length=env_len)
        if (env_len > 0) trace_point_vars = trim(env_value(:env_len))

        trace_initialized = .true.

    end subroutine s_initialize_trace

    !> Emit a live call-trace line when MFC_TRACE is enabled.
    subroutine s_trace_enter(name)

        character(len=*), intent(in) :: name

        call s_initialize_trace()

        if (.not. trace_enabled) return
        if (trace_point_enabled .and. trace_point_depth <= 0) return

        write (output_unit, '(A,A)') 'TRACE ', trim(name)
        call flush (output_unit)

    end subroutine s_trace_enter

    !> Emit a live call-site trace when the current trace scope is active.
    subroutine s_trace_call(name)

        character(len=*), intent(in) :: name

        call s_initialize_trace()

        if (.not. trace_enabled) return
        if (trace_point_enabled .and. trace_point_depth <= 0) return

        write (output_unit, '(A,A)') 'TRACE ', trim(name)
        call flush (output_unit)

    end subroutine s_trace_call

    !> Emit a call-site trace even when point tracing has not entered a cell loop yet.
    subroutine s_trace_global_call(name)

        character(len=*), intent(in) :: name

        call s_initialize_trace()

        if (.not. trace_enabled) return

        write (output_unit, '(A,A)') 'TRACE ', trim(name)
        call flush (output_unit)

    end subroutine s_trace_global_call

    !> Enable nested routine-entry tracing while a call at MFC_TRACE_POINT executes.
    subroutine s_trace_point_begin(j, k, l, vars, mid_j, mid_k, mid_l)

        integer, intent(in)          :: j
        integer, intent(in)          :: k
        integer, intent(in)          :: l
        character(len=*), intent(in) :: vars
        integer, intent(in)          :: mid_j
        integer, intent(in)          :: mid_k
        integer, intent(in)          :: mid_l
        integer                      :: target_j
        integer                      :: target_k
        integer                      :: target_l

        call s_initialize_trace()

        if (.not. trace_enabled) return
        if (.not. trace_point_enabled) return
        if (len_trim(trace_point_vars) > 0 .and. trim(trace_point_vars) /= 'any') then
            if (trim(vars) /= trim(trace_point_vars)) return
        end if
        if (trace_point_middle) then
            target_j = mid_j
            target_k = mid_k
            target_l = mid_l
        else
            target_j = trace_j
            target_k = trace_k
            target_l = trace_l
        end if

        if (j /= target_j .or. k /= target_k .or. l /= target_l) return

        trace_point_depth = trace_point_depth + 1
        call c_trace_point_begin()

    end subroutine s_trace_point_begin

    !> Disable nested point tracing after a grid-point call returns.
    subroutine s_trace_point_end

        call s_initialize_trace()

        if (.not. trace_enabled) return
        if (.not. trace_point_enabled) return
        if (trace_point_depth > 0) then
            call c_trace_point_end()
            trace_point_depth = trace_point_depth - 1
        end if

    end subroutine s_trace_point_end

end module m_trace
