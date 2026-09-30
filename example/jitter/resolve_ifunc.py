import logging
from argparse import ArgumentParser

from miasm.analysis.binary import Container, ContainerELF
from miasm.analysis.machine import Machine
from miasm.core.locationdb import LocationDB
from miasm.jitter.loader.elf import get_ifuncs, run_ifunc_resolvers_copy
from miasm.loader.elf import ET_EXEC


def code_sentinelle(jitter):
    jitter.running = False
    jitter.pc = 0
    return False

def prepare(run_ifuncs: bool):
    loc_db = LocationDB()

    myjit = Machine("x86_64").jitter(loc_db, args.jitter)
    myjit.init_stack()
    base_addr = 0x400000

    with open(args.filename, 'rb') as f:
        elf: ContainerELF = Container.from_stream(f, addr=base_addr, loc_db=loc_db, vm=myjit.vm, apply_reloc=True, run_ifuncs=run_ifuncs)
        assert isinstance(elf, ContainerELF)
    if elf.executable.Ehdr.type == ET_EXEC:
        # static executables can't be rebased
        base_addr = 0

    return (base_addr, loc_db, elf, myjit)

def launch(jitter):
    jitter.push_uint64_t(0x1337beef)
    jitter.add_breakpoint(0x1337beef, code_sentinelle)

    if args.verbose >= 2:
        jitter.set_trace_log(True, True)
    run_at = jitter.lifter.loc_db.get_name_offset("intermediate")
    assert run_at is not None
    jitter.run(run_at)
    return jitter.get_c_str(jitter.cpu.RAX)

if __name__ == "__main__":
    parser = ArgumentParser(description="x86 ELF ifunc relocs (e.g. apsamples/ifunc)")
    parser.add_argument("filename", help="ELF to apply (ifunc) relocs to")
    parser.add_argument("-j", "--jitter",
                        help="Jitter engine (default is 'gcc')",
                        default="gcc")
    parser.add_argument("--verbose", "-v", action="count",
                        help="Verbose mode (-v activates debug logging, -vv adds jitter tracing)",
                        default=0)
    args = parser.parse_args()

    # setup logging
    log = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)-8s]: %(message)s")
    ch.setFormatter(formatter)
    if args.verbose >= 1:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.WARNING)
    log.addHandler(ch)


    # 1. Run ifunc with default resolver behavior

    _, _, _, jitter = prepare(run_ifuncs=True)
    res1 = launch(jitter)
    log.info(f"Default ifunc resolving returned {res1}.")
    assert res1 == "zglorg"


    # 2. Run ifunc with alternate resolver behavior

    base_addr, loc_db, elf, jitter = prepare(run_ifuncs=False)

    # The ifunc resolver in ../samples/ifunc uses .bss variable `use_func2`
    # to know whether to redirect to `func1` or `func2` when we execute `func`
    # which we proceed to set to `true`
    use_func2_addr = loc_db.get_name_offset("use_func2")
    if use_func2_addr is not None and jitter.vm.is_mapped(use_func2_addr + base_addr, 1):
        jitter.vm.set_mem(use_func2_addr + base_addr, b'\x01')
        log.info(f"Set symbol use_func2 (@0x{use_func2_addr + base_addr:x}) to 1")
    else:
        if use_func2_addr is None:
            raise ValueError("symbol use_func2 was not found")
        else:
            raise ValueError(f"use_func2 exists but doesn't seem to be loaded inside our vm ({use_func2_addr+base_addr=:x})\n{jitter.vm}")

    # we can then run our infunc resolver
    ifunc_resolvers = get_ifuncs(elf.executable, base_addr, with_syms=False)
    run_ifunc_resolvers_copy(ifunc_resolvers, elf.executable, jitter.vm, loc_db)

    res2 = launch(jitter)
    log.info(f"Forcing ifunc resolving to func2 returned {res2}.")
    assert res2 == "bloups"
