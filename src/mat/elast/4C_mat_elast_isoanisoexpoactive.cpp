// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
// eigene 4C Materialklasse IsoAnisoExpoActive fuer aktive Muskelmaterialien

#include "4C_mat_elast_isoanisoexpoactive.hpp"
#include <cmath>
#include <iostream>  
#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_FADmatrix_utils.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

Mat::Elastic::PAR::IsoAnisoExpoActive::IsoAnisoExpoActive(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : ParameterAniso(matdata),
      gamma_(matdata.parameters.get<double>("GAMMA")),
      init_(matdata.parameters.get<int>("INIT")),
      adapt_angle_(matdata.parameters.get<bool>("ADAPT_ANGLE")),
      mue_smc_(matdata.parameters.get<double>("MUE_SMC")),
      urs0_(matdata.parameters.get<double>("URS0")),
      a0_contraction_(matdata.parameters.get<double>("A0")),
      v_(matdata.parameters.get<double>("V")),
      delta_l_(matdata.parameters.get<double>("DELTA_L")),
      t0_(matdata.parameters.get<double>("T0"))
{
}

Mat::Elastic::IsoAnisoExpoActive::IsoAnisoExpoActive(
    Mat::Elastic::PAR::IsoAnisoExpoActive* params)
    : params_(params),
      anisotropy_extension_(params_->init_, params_->gamma_, params_->adapt_angle_ != 0,
          params_->structural_tensor_strategy(), {0})
{
  d_p_iact_ = 0.0;
  lambdaact_ = 1.0;
  z_ref_ = 0.0;
  anisotropy_extension_.register_needed_tensors(
      FiberAnisotropyExtension<1>::FIBER_VECTORS | FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR);
}

void Mat::Elastic::IsoAnisoExpoActive::register_anisotropy_extensions(Mat::Anisotropy& anisotropy)
{
  anisotropy.register_anisotropy_extension(anisotropy_extension_);
}

void Mat::Elastic::IsoAnisoExpoActive::pack_summand(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, lambdaact_);
  add_to_pack(data, z_ref_);
  anisotropy_extension_.pack_anisotropy(data);
}

void Mat::Elastic::IsoAnisoExpoActive::unpack_summand(Core::Communication::UnpackBuffer& buffer)
{
  extract_from_pack(buffer, lambdaact_);
  extract_from_pack(buffer, z_ref_);
  anisotropy_extension_.unpack_anisotropy(buffer);

  d_p_iact_ = evaluated_psi_active();
}

void Mat::Elastic::IsoAnisoExpoActive::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  lambdaact_ = 1.0;
  d_p_iact_ = evaluated_psi_active();

  z_ref_ = 0.0;
  if (fibers.element_fibers.size() > 1)
  {
    const double zx = fibers.element_fibers[1](0);
    const double zy = fibers.element_fibers[1](1);
    z_ref_ = (std::abs(zy) > 1e-12) ? zx / zy : 0.0;   // z = n_x / n_y, normierungs-fest
  }
}

void Mat::Elastic::IsoAnisoExpoActive::add_strain_energy(double& psi,
    const Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp,
    const int eleGID)
{
  // Psi^smc = 0.5 * mu_smc * [ sqrt(I4) + u_rs - 1 ]^4
  // At z=0: u_rs = urs0 + a0_contraction
  const double u_rs = params_->urs0_ + params_->a0_contraction_;
  const double f    = lambdaact_ + u_rs - 1.0;  // lambdaact_ = sqrt(I4) = lambda_f
  psi += 0.5 * params_->mue_smc_ * f * f * f * f;
}

template <typename T>
void Mat::Elastic::IsoAnisoExpoActive::evaluate_func(
    T& psi, Core::LinAlg::SymmetricTensor<T, 3, 3> const& rcg, const int gp, const int eleGID) const
{
  // Active-only material: no passive anisotropic strain energy contribution.
  (void)psi;
  (void)rcg;
  (void)gp;
  (void)eleGID;
}

template <typename T>
void Mat::Elastic::IsoAnisoExpoActive::evaluate_active_stress_cmat_aniso(
    Core::LinAlg::SymmetricTensor<T, 3, 3> const& CM,
    Core::LinAlg::SymmetricTensor<T, 3, 3, 3, 3>& cmat,
    Core::LinAlg::SymmetricTensor<T, 3, 3>& stress, const int gp, const int eleGID) const
{
  T lambda_sq = Core::LinAlg::ddot(CM, anisotropy_extension_.get_structural_tensor(gp, 0));
  T dPIact_T = d_p_iact_;
  stress = T(dPIact_T / lambda_sq) * anisotropy_extension_.get_structural_tensor(gp, 0);
  cmat = T(-2.0 * dPIact_T / (lambda_sq * lambda_sq)) *
         Core::LinAlg::dyadic(anisotropy_extension_.get_structural_tensor(gp, 0),
             anisotropy_extension_.get_structural_tensor(gp, 0));
}

void Mat::Elastic::IsoAnisoExpoActive::add_active_stress_cmat_aniso(
    Core::LinAlg::SymmetricTensor<double, 3, 3> const& CM,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress, const int gp, const int eleGID,
    const double totaltime) const
{
  const auto& M = anisotropy_extension_.get_structural_tensor(gp, 0);

  const double lambda_sq = Core::LinAlg::ddot(CM, M);
  const double lambda    = std::sqrt(lambda_sq);  // = sqrt(I4) je GP = Faserdehnung

  // --- Wandernde Aktivierungswelle: u_rs(z, t) (Peirlinck et al.) ---
  // urs0                    for   t - t0 <= z/v
  // urs0 + a0 * cos^2(...)  for   z/v < t - t0 <= (z + DELTA_L)/v
  // urs0                    for   (z + DELTA_L)/v <= t - t0
  double u_rs = params_->urs0_;
  if (params_->v_ > 0.0 && params_->delta_l_ > 0.0)
  {
    // xi = wie weit die Wellenfront die Materialposition z schon passiert hat (Wellenkoordinate)
    const double xi = params_->v_ * (totaltime - params_->t0_) - z_ref_;
    //                = v · (t − t0)              − z
    if (xi > 0.0 && xi < params_->delta_l_) //abfangen der stelle wo die Welle grade aktiv ist
    {
      // glatter Puls: 0 an den Segmenträndern, Maximum in der Segmentmitte
      // (cos^2 zentriert auf xi = DELTA_L/2; ggf. gegen Fig. im Paper abgleichen)
      const double c = std::cos(M_PI / params_->delta_l_ * (xi - 0.5 * params_->delta_l_));
      u_rs += params_->a0_contraction_ * c * c;
    }
  }
  else
  {
    // Fallback ohne Wellenparameter: konstante Aktivierung (Baustelle-1-Verhalten)
    u_rs += params_->a0_contraction_;
  }

  // ===================== TEMP-DEBUG (danach wieder raus) =====================
 // TEMP-DEBUG: unabhängig von z, feuert für ein paar Muskelelemente
if (gp == 0 && eleGID % 200 == 0)
{
  std::cout << "[urs-debug] ele=" << eleGID
            << "  t="   << totaltime
            << "  z="   << z_ref_
            << "  u_rs=" << u_rs << std::endl;
}
  // ==========================================================================


  const double f       = lambda + u_rs - 1.0;
  const double dPIact  = 2.0 * params_->mue_smc_ * f * f * f;  // dPsi_smc/dlambda

  stress += dPIact / lambda_sq * M;

  // Konsistenter Materialtangens dS/dE = 2*dS/dC (u_rs haengt nicht von C ab):
  //   Term 1: Ableitung von dPIact nach lambda -> +6*mue*f^2 / (lambda*I4)
  //   Term 2: geometrischer Anteil d(1/I4)/dC  -> -2*dPIact / I4^2
  const double dSdE = 6.0 * params_->mue_smc_ * f * f / (lambda * lambda_sq) -
                      2.0 * dPIact / (lambda_sq * lambda_sq);
  cmat += dSdE * Core::LinAlg::dyadic(M, M);
}

void Mat::Elastic::IsoAnisoExpoActive::evaluate_first_derivatives_aniso(
    Core::LinAlg::Matrix<2, 1>& dPI_aniso, Core::LinAlg::SymmetricTensor<double, 3, 3> const& rcg,
    int gp, int eleGID)
{
  (void)rcg;
  (void)gp;
  (void)eleGID;
  dPI_aniso(0) = 0.0;
}

void Mat::Elastic::IsoAnisoExpoActive::evaluate_second_derivatives_aniso(
    Core::LinAlg::Matrix<3, 1>& ddPII_aniso,
    Core::LinAlg::SymmetricTensor<double, 3, 3> const& rcg, int gp, int eleGID)
{
  (void)rcg;
  (void)gp;
  (void)eleGID;
  ddPII_aniso(0) = 0.0;
}

template <typename T>
void Mat::Elastic::IsoAnisoExpoActive::get_derivatives_aniso(
    Core::LinAlg::Matrix<2, 1, T>& dPI_aniso, Core::LinAlg::Matrix<3, 1, T>& ddPII_aniso,
    Core::LinAlg::Matrix<4, 1, T>& dddPIII_aniso,
    const Core::LinAlg::SymmetricTensor<T, 3, 3>& rcg, const int gp, const int eleGID) const
{
  (void)rcg;
  (void)gp;
  (void)eleGID;
  dPI_aniso(0) = T(0.0);
  ddPII_aniso(0) = T(0.0);
  dddPIII_aniso(0) = T(0.0);
};

void Mat::Elastic::IsoAnisoExpoActive::add_stress_aniso_principal(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    const Teuchos::ParameterList& params, const int gp, const int eleGID)
{
  (void)rcg;
  (void)cmat;
  (void)stress;
  (void)params;
  (void)gp;
  (void)eleGID;
  // No passive anisotropic contribution for active-only muscle material.
}

void Mat::Elastic::IsoAnisoExpoActive::get_fiber_vecs(
    std::vector<Core::LinAlg::Tensor<double, 3>>& fibervecs) const
{
  fibervecs.push_back(anisotropy_extension_.get_fiber(BaseAnisotropyExtension::GPDEFAULT, 0));
}

void Mat::Elastic::IsoAnisoExpoActive::set_fiber_vecs(const double newgamma,
    const Core::LinAlg::Tensor<double, 3, 3>& locsys,
    const Core::LinAlg::Tensor<double, 3, 3>& defgrd)
{
  anisotropy_extension_.set_fiber_vecs(newgamma, locsys, defgrd);
}

double Mat::Elastic::IsoAnisoExpoActive::evaluated_psi_active() const
{
  // dPsi/d(lambda) = 2 * mu_smc * f^3
  // (Chain rule: dPsi/dI4 * dI4/dlambda = dPsi/dI4 * 2*lambda,
  //  but active stress needs dPsi/dlambda directly)
  const double u_rs = params_->urs0_ + params_->a0_contraction_;
  const double f    = lambdaact_ + u_rs - 1.0;
  return 2.0 * params_->mue_smc_ * f * f * f;
}

// explicit instantiation of template functions
template void Mat::Elastic::IsoAnisoExpoActive::get_derivatives_aniso<double>(
    Core::LinAlg::Matrix<2, 1, double>&, Core::LinAlg::Matrix<3, 1, double>&,
    Core::LinAlg::Matrix<4, 1, double>&, Core::LinAlg::SymmetricTensor<double, 3, 3> const&, int,
    const int) const;
template void Mat::Elastic::IsoAnisoExpoActive::get_derivatives_aniso<FAD>(
    Core::LinAlg::Matrix<2, 1, FAD>&, Core::LinAlg::Matrix<3, 1, FAD>&,
    Core::LinAlg::Matrix<4, 1, FAD>&, Core::LinAlg::SymmetricTensor<FAD, 3, 3> const&, int,
    const int) const;
template void Mat::Elastic::IsoAnisoExpoActive::evaluate_active_stress_cmat_aniso<double>(
    Core::LinAlg::SymmetricTensor<double, 3, 3> const&,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>&,
    Core::LinAlg::SymmetricTensor<double, 3, 3>&, int, const int) const;
template void Mat::Elastic::IsoAnisoExpoActive::evaluate_active_stress_cmat_aniso<FAD>(
    Core::LinAlg::SymmetricTensor<FAD, 3, 3> const&,
    Core::LinAlg::SymmetricTensor<FAD, 3, 3, 3, 3>&,
    Core::LinAlg::SymmetricTensor<FAD, 3, 3>&, int, const int) const;
template void Mat::Elastic::IsoAnisoExpoActive::evaluate_func<double>(
    double&, Core::LinAlg::SymmetricTensor<double, 3, 3> const&, int, const int) const;
template void Mat::Elastic::IsoAnisoExpoActive::evaluate_func<FAD>(
    FAD&, Core::LinAlg::SymmetricTensor<FAD, 3, 3> const&, int, const int) const;

FOUR_C_NAMESPACE_CLOSE
