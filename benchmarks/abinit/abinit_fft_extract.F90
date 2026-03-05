subroutine sg_fft_cc(fftcache,n1,n2,n3,nd1,nd2,nd3,ndat,isign,arr,ftarr)

!Arguments ------------------------------------
!scalars
 integer,intent(in) :: fftcache,n1,n2,n3,nd1,nd2,nd3,ndat,isign
!arrays
 real(dp),intent(inout) :: arr(2,nd1*nd2*nd3*ndat)
 real(dp),intent(inout) :: ftarr(2,nd1*nd2*nd3*ndat)

!Local variables-------------------------------
!scalars
 integer :: idat,start

! *************************************************************************

 do idat=1,ndat
   start = 1 + (idat-1)*nd1*nd2*nd3
   call fft_cc_one_nothreadsafe(fftcache,nd1,nd2,nd3,n1,n2,n3,arr(1,start),ftarr(1,start),real(isign,kind=dp))
 end do

end subroutine sg_fft_cc

! ========== sg_fftpx (lines 843-1529, 687 lines total) ==========
subroutine sg_fftpx(fftcache,mfac,mg,mgfft,nd1,nd2,nd3,n2,n3,&
&    z,zbr,trig,aft,now,bef,ris,ind,ic,gbound)

!Arguments ------------------------------------
!Dimensions of aft, now, bef, ind, and trig should agree with
!those in subroutine ctrig.
!scalars
 integer,intent(in) :: fftcache,ic,mfac,mg,mgfft,n2,n3,nd1,nd2,nd3
 real(dp),intent(in) :: ris
!arrays
 integer,intent(in) :: aft(mfac),bef(mfac),gbound(2*mgfft+4),ind(mg),now(mfac)
 real(dp),intent(in) :: trig(2,mg)
 real(dp),intent(inout) :: z(2,nd1,nd2,nd3)
 real(dp),intent(inout) :: zbr(2,nd1,nd2,nd3) !vz_i

!Local variables-------------------------------
!scalars
 integer :: g2,g2max,g2min,g3,g3max,g3min,gg3,i,ia,ib,igb,ihalfy,indx,j
 integer :: len3,lot,lowlim,ma,mb,ntb,upplim
!no_abirules
 real(dp),parameter :: &
& cos2=0.3090169943749474d0,&   !cos(2.d0*pi/5.d0)
& cos4=-0.8090169943749474d0,&  !cos(4.d0*pi/5.d0)
& sin42=0.6180339887498948d0    !sin(4.d0*pi/5.d0)/sin(2.d0*pi/5.d0)
 real(dp) :: bb,cr2,cr2s,cr3,cr3p,cr4,cr5,ct2,ct3,ct4,ct5,&
& factor,r,r1,r2,r25,r3,r34,r4,r5,s,sin2,s1,s2,s25,s3,s34,s4,s5

! *************************************************************************

 g3min=gbound(1)
 g3max=gbound(2)
 igb=3
 len3=g3max-g3min+1


!Do x transforms in blocks of size "lot" which is set by how
!many x transform arrays (of size nd1 each) fit into the nominal
!cache size "fftcache".
!Loop over blocks in the loop below.

 factor=0.75d0
 lot=(fftcache*factor*1000d0)/(nd1*8*2)
 if(lot.lt.1) lot=1
!Express loop over y, z in terms of separate z and y loops

!$OMP PARALLEL DO DEFAULT(PRIVATE)&
!$OMP SHARED(aft,bef,gbound,g3max,ic,ind,len3,lot)&
!$OMP SHARED(n2,n3,nd2,now,ris,trig,z,zbr)
 do gg3=1,len3

   if (gg3<=g3max+1) then
     g3=gg3
   else
!    wrap around for negative gg3
     g3=gg3-len3+n3
   end if

   igb=gg3*2+1
   g2min=gbound(igb)
   g2max=gbound(igb+1)


! ========== sg_ffty (lines 2288-2917, 630 lines total) ==========
subroutine sg_ffty(fftcache,mfac,mg,nd1,nd2,nd3,n1i,n1,n3i,n3,&
&          z,zbr,trig,aft,now,bef,ris,ind,ic)

!Arguments ------------------------------------
!Dimensions of aft, now, bef, ind, and trig should agree with
!those in subroutine ctrig.
!scalars
 integer,intent(in) :: fftcache,ic,mfac,mg,n1,n1i,n3,n3i,nd1,nd2,nd3
 real(dp),intent(in) :: ris
!arrays
 integer,intent(in) :: aft(mfac),bef(mfac),ind(mg),now(mfac)
 real(dp),intent(in) :: trig(2,mg)
 real(dp),intent(inout) :: z(2,nd1,nd2,nd3),zbr(2,nd1,nd2,nd3)

!Local variables-------------------------------
!scalars
 integer :: i,ia,ib,indx,j1,j2,ntb
 real(dp),parameter :: cos2=0.3090169943749474d0   !cos(2.d0*pi/5.d0)
 real(dp),parameter :: cos4=-0.8090169943749474d0  !cos(4.d0*pi/5.d0)
 real(dp),parameter :: sin42=0.6180339887498948d0  !sin(4.d0*pi/5.d0)/sin(2.d0*pi/5.d0)
 real(dp) :: bb,cr2,cr2s,cr3,cr3p,cr4,cr5,ct2,ct3,ct4,ct5
 real(dp) :: r,r1,r2,r25,r3,r34,r4,r5,s,sin2,s1,s2,s25,s3,s34,s4,s5

! *************************************************************************

 if (fftcache<0) then
   ABI_ERROR('fftcache must be positive')
 end if

!Outer loop over z planes (j2)--note range from n3i to n3

!$OMP PARALLEL DO DEFAULT(PRIVATE) SHARED(aft,bef,ic,ind,n1,n1i,n3,n3i,now,ris,trig,z,zbr)
 do j2=n3i,n3

!  Direct transformation
   do i=1,ic-1
     ntb=now(i)*bef(i)

!    Treat radix 4
     if (now(i)==4) then
       ia=0

!      First step of radix 4
       do ib=1,bef(i)
!        Inner loop over all x values (j1) -- note range from n1i to n1
!        y transform is performed for this range of x values repeatedly
!        below

         do j1=n1i,n1
           r4=z(1,j1,ia*ntb+3*bef(i)+ib,j2)
           s4=z(2,j1,ia*ntb+3*bef(i)+ib,j2)
           r3=z(1,j1,ia*ntb+2*bef(i)+ib,j2)
           s3=z(2,j1,ia*ntb+2*bef(i)+ib,j2)
           r2=z(1,j1,ia*ntb+bef(i)+ib,j2)
           s2=z(2,j1,ia*ntb+bef(i)+ib,j2)
           r1=z(1,j1,ia*ntb+ib,j2)
           s1=z(2,j1,ia*ntb+ib,j2)

           r=r1 + r3
           s=r2 + r4
           z(1,j1,ia*ntb+ib,j2) = r + s

! ========== sg_fftrisc (lines 3851-3909) ==========
subroutine sg_fftrisc(cplex,denpot,fofgin,fofgout,fofr,gboundin,gboundout,istwf_k,&
& kg_kin,kg_kout,mgfft,ndat,ngfft,npwin,npwout,n4,n5,n6,option,weight_r, weight_i)

!Arguments ------------------------------------
!scalars
 integer,intent(in) :: cplex,istwf_k,mgfft,n4,n5,n6,ndat,npwin,npwout,option
 real(dp),intent(in) :: weight_i,weight_r
!arrays
 integer,intent(in) :: gboundin(2*mgfft+8,2),gboundout(2*mgfft+8,2)
 integer,intent(in) :: kg_kin(3,npwin),kg_kout(3,npwout),ngfft(18)
 real(dp),intent(in) :: fofgin(2,npwin*ndat)
 real(dp),intent(inout) :: denpot(cplex*n4*n5*n6),fofr(2,n4*n5*n6*ndat)
 real(dp),intent(out) :: fofgout(2,npwout*ndat)

!Local variables-------------------------------
!scalars
 integer :: idat,fofgin_p,fofr_p,fofgout_p
!arrays
 real(dp) :: dum_fofgin(0,0),dum_fofr(0,0),dum_fofgout(0,0)

! *************************************************************************

 do idat=1,ndat
   fofgin_p = 1 + (idat-1) * npwin
   fofr_p = 1 + (idat - 1) * n4*n5*n6
   fofgout_p = 1 + (idat-1) * npwout

   select case (option)
   case (0)
     call fftrisc_one_nothreadsafe(&
&      cplex,denpot,fofgin(1,fofgin_p),dum_fofgout,fofr(1,fofr_p),&
&      gboundin,gboundout,istwf_k,&
&      kg_kin,kg_kout,mgfft,ngfft,npwin,npwout,n4,n5,n6,option,weight_r,weight_i)

   case (1)
     ! Don't know why but fofr is not touched by this option.
     call fftrisc_one_nothreadsafe(&
&      cplex,denpot,fofgin(1,fofgin_p),dum_fofgout,dum_fofr,&
&      gboundin,gboundout,istwf_k,&
&      kg_kin,kg_kout,mgfft,ngfft,npwin,npwout,n4,n5,n6,option,weight_r,weight_i)

   case (2)
     call fftrisc_one_nothreadsafe(&
&      cplex,denpot,fofgin(1,fofgin_p),fofgout(1,fofgout_p),dum_fofr,&
&      gboundin,gboundout,istwf_k,&
&      kg_kin,kg_kout,mgfft,ngfft,npwin,npwout,n4,n5,n6,option,weight_r,weight_i)

   case (3)
     call fftrisc_one_nothreadsafe(&
&      cplex,denpot,dum_fofgin,fofgout(1,fofgout_p),fofr(1,fofr_p),&
&      gboundin,gboundout,istwf_k,&
&      kg_kin,kg_kout,mgfft,ngfft,npwin,npwout,n4,n5,n6,option,weight_r,weight_i)

   case default
      ABI_ERROR("Wrong option")
   end select
 end do

end subroutine sg_fftrisc
